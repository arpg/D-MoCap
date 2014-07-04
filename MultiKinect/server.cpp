#include <Eigen/Eigen>
#include <sophus/se3.hpp>

#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <pangolin/glvbo.h>

#include <SceneGraph/SceneGraph.h>

#include <kangaroo/kangaroo.h>
#include <kangaroo/BoundedVolume.h>
#include <kangaroo/MarchingCubes.h>
#include <kangaroo/extra/ImageSelect.h>
#include <kangaroo/extra/BaseDisplayCuda.h>
#include <kangaroo/extra/DisplayUtils.h>
#include <kangaroo/extra/Handler3dGpuDepth.h>
#include <kangaroo/extra/SavePPM.h>
#include <kangaroo/extra/SaveMeshlab.h>

#ifdef HAVE_CVARS
#include <kangaroo/extra/CVarHelpers.h>
#endif // HAVE_CVARS

#include <opencv2/opencv.hpp>
#include <unistd.h>

#include <HAL/Utils/GetPot>
#include <HAL/Utils/TicToc.h>
#include <HAL/Camera/CameraDevice.h>
#include <calibu/Calibu.h>
#include <CVars/CVar.h>

#include <Node/Node.h>
#include <string>

using namespace pangolin;

int main( int argc, char* argv[] )
{
  node::node server;
  server.set_verbosity(9);
  server.init("MKServer");

//  std::vector< node::node > nodes;
  // Initialise window
  View& container = SetupPangoGLWithCuda(1024, 768);
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  GetPot clArgs(argc, argv);
  const int num_nodes = atoi(clArgs.follow("0", "-nodes").c_str());

  for(int ii = 0; ii < num_nodes; ii++) {
    std::string topic;

  }

  const int originalWidth = camera.Width();
  const int originalHeight = camera.Height();
  std::cout << "- Image Dimensions: " << originalWidth << "x" << originalHeight
            << std::endl;

  const unsigned int imageLevel = level;
  const unsigned int imageWidth = originalWidth >> level;
  const unsigned int imageHeight = originalHeight >> level;

  if (level != 0) {
    std::cout << "- Images exceed maximum dimensions. Using level: " <<
                 imageLevel << std::endl;
    std::cout << "- NEW Image Dimensions: " << imageWidth << "x" << imageHeight
              << std::endl;
  }

  ///----- Load camera model.
  calibu::CameraRig rig;
  if (!camera.GetDeviceProperty(hal::DeviceDirectory).empty()) {
    std::cout<<"Loaded camera: "<<camera.GetDeviceProperty(hal::DeviceDirectory) + '/' + clArgs.follow("cameras.xml", "-cmod")<<std::endl;
    rig = calibu::ReadXmlRig(camera.GetDeviceProperty(hal::DeviceDirectory) + '/' + clArgs.follow("cameras.xml", "-cmod"));
  }
  else {
    rig = calibu::ReadXmlRig(clArgs.follow("cameras.xml", "-cmod"));
  }

  // Force vision convention.
  rig = calibu::ToCoordinateConvention(rig, calibu::RdfVision);

  cudaClass cuda;
  Eigen::Matrix3f K, Kinv;
  K = rig.cameras[0].camera.K().cast<float>();
  Kinv = K.inverse();


  ///----- Init aux variables and dense tracker.

  const bool use_colour = false;

  std::vector<pangolin::Image<unsigned char> > imgs;
  const int w = imageWidth;
  const int h = imageHeight;

  const int MaxLevels = 3;
  const int its[] = {1,0,2,3};

  const roo::ImageIntrinsics K(KL(0, 0),KL(1, 1), KL(0, 2), KL(1, 2) );

  const double knear = 0.01;
  const double kfar = 1.0;

  const float volrad = 0.250;
  const int volres = 256;

  roo::BoundingBox reset_bb(make_float3(-volrad,-volrad,knear), make_float3(volrad,volrad,knear+2*volrad));
  //    roo::BoundingBox reset_bb(make_float3(-volrad,-volrad,-volrad), make_float3(volrad,volrad,volrad));

#ifdef HAVE_CVARS
  CVarUtils::AttachCVar<roo::BoundingBox>("BoundingBox", &reset_bb);
#endif // HAVE_CVARS

  // Camera (rgb) to depth
  roo::Image<float, roo::TargetDevice, roo::Manage> dKinectMeters(w,h);
  roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> kin_d(w,h);
  roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> kin_v(w,h);
  roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> kin_n(w,h);
  roo::Image<float4, roo::TargetDevice, roo::Manage>  dDebug(w,h);

  roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> ray_i(w,h);
  roo::Pyramid<float, MaxLevels, roo::TargetDevice, roo::Manage> ray_d(w,h);
  roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_n(w,h);
  roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_v(w,h);
  roo::Pyramid<float4, MaxLevels, roo::TargetDevice, roo::Manage> ray_c(w,h);
  roo::BoundedVolume<roo::SDF_t, roo::TargetDevice, roo::Manage> vol(volres,volres,volres,reset_bb);
  roo::BoundedVolume<float, roo::TargetDevice, roo::Manage> colorVol(volres,volres,volres,reset_bb);

  std::vector<std::unique_ptr<KinectKeyframe> > keyframes;
  roo::Mat<roo::ImageKeyframe<uchar3>,10> kfs;

  SceneGraph::GLSceneGraph glgraph;
  SceneGraph::GLAxis glcamera(0.1);
  SceneGraph::GLAxisAlignedBox glboxfrustum;
  SceneGraph::GLAxisAlignedBox glboxvol;

  glboxvol.SetBounds(roo::ToEigen(vol.bbox.Min()), roo::ToEigen(vol.bbox.Max()) );
  glgraph.AddChild(&glcamera);
  glgraph.AddChild(&glboxvol);
  glgraph.AddChild(&glboxfrustum);

  pangolin::OpenGlRenderState s_cam(
        ProjectionMatrixRDF_TopLeft(w,h,K.fu,K.fv,K.u0,K.v0,0.1,1000),
        ModelViewLookAtRDF(0,0,-2,0,0,0,0,-1,0)
        );

  Var<bool> reset("ui.reset", true, false);
  Var<bool> run("ui.run", true, true);

  Var<bool> viewonly("ui.view only", true, true);
  Var<bool> fuse("ui.fuse", false, true);
  Var<float> max_rmse("ui.Max RMSE",0.10,0,0.5);
  Var<float> rmse("ui.RMSE",0);

  Var<int> show_level("ui.Show Level", 0, 0, MaxLevels-1);

  // TODO: This needs to be a function of the inverse depth
  Var<int>   biwin("ui.size",3, 1, 20);      // Bilateral filter
  Var<float> bigs("ui.gs",1.5, 1E-3, 5);     // Bilateral filter
  Var<float> icp_c("ui.icp c",0.1, 1E-3, 1);
  Var<float> bigr("ui.gr",0.1, 1E-6, 0.2);   // Bilateral filter
  roo::Image<unsigned char, roo::TargetDevice, roo::Manage> dScratch(w*sizeof(roo::LeastSquaresSystem<float,12>),h);
  Var<float> trunc_dist_factor("ui.trunc vol factor",2, 1, 4);

  Var<float> max_w("ui.max w", 1000, 1E-2, 1E3);
  Var<float> mincostheta("ui.min cos theta", 0.1, 0, 1);
  Var<float> rgb_fl("ui.RGB focal length", 535.7,400,600);

  ActivateDrawPyramid<float,MaxLevels> adrayimg(ray_i, GL_LUMINANCE32F_ARB, true, true);
  ActivateDrawPyramid<float4,MaxLevels> adraycolor(ray_c, GL_RGBA32F, true, true);
  ActivateDrawPyramid<float4,MaxLevels> adraynorm(ray_n, GL_RGBA32F, true, true);
  //    ActivateDrawPyramid<float,MaxLevels> addepth( kin_d, GL_LUMINANCE32F_ARB, false, true);
  ActivateDrawPyramid<float4,MaxLevels> adnormals( kin_n, GL_RGBA32F_ARB, false, true);
  ActivateDrawImage<float4> addebug( dDebug, GL_RGBA32F_ARB, false, true);

  Handler3DDepth<float,roo::TargetDevice> rayhandler(ray_d[0], s_cam, AxisNone);
  SetupContainer(container, 4, (float)w/h);
  container[0].SetDrawFunction(std::ref(adrayimg))
      .SetHandler(&rayhandler);
  container[1].SetDrawFunction(SceneGraph::ActivateDrawFunctor(glgraph, s_cam))
      .SetHandler( new Handler3D(s_cam, AxisNone) );
  container[2].SetDrawFunction(std::ref(use_colour?adraycolor:adraynorm))
      .SetHandler(&rayhandler);
  container[3].SetDrawFunction(std::ref(adnormals));

  float* depthBuffer = (float *) malloc(imageWidth*imageHeight*sizeof(float));
  SceneGraph::ImageView depthView;
  SceneGraph::ImageView currentView;
  DepthHandler depthHandler(imageWidth, imageHeight, depthBuffer);
  depthView.SetHandler( &depthHandler );
  depthView.ToggleShow();
  container.AddDisplay( depthView );
  container.AddDisplay( currentView );

  Sophus::SE3d T_wl;

  bool bStep = false;
  pangolin::RegisterKeyPressCallback(' ', [&viewonly]() { viewonly=!viewonly;} );
  pangolin::RegisterKeyPressCallback('l', [&vol,&viewonly]() {LoadPXM("save.vol", vol); viewonly = true;} );
  //    pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol,&keyframes,&rgb_fl,w,h]() {SavePXM("save.vol", vol); SaveMeshlab(vol,keyframes,rgb_fl,rgb_fl,w/2,h/2); } );
  pangolin::RegisterKeyPressCallback('s', [&vol,&colorVol]() {roo::SaveMesh("mesh",vol,colorVol); } );
  pangolin::RegisterKeyPressCallback('d', [&]() { DumpDepthMap(depthBuffer, KLinv, imageWidth, imageHeight); } );
  pangolin::RegisterKeyPressCallback('/', [&]() {bStep = !bStep; } );
  //    pangolin::RegisterKeyPressCallback('s', [&vol]() {SavePXM("save.vol", vol); } );

  Sophus::SE3d current_pose, rel, start, last;
  bool started = false;
  reset = true;
  Eigen::Matrix<double, 3, 1> sdf_trans;
  sdf_trans << 0, 0, 0;
  Sophus::SE3d Tws = Sophus::SE3d(Eigen::Matrix3d::Identity(), sdf_trans);

  bool first;

  for(long frame=-1; !pangolin::ShouldQuit();)
  {
    const bool go = !viewonly && (frame==-1 || run);

    const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());

    if(pangolin::Pushed(reset)) {
      T_wl = Sophus::SE3d();
      fuse = false;
      started = false;

      vol.bbox = reset_bb;
      roo::SdfReset(vol, std::numeric_limits<float>::quiet_NaN() );
      keyframes.clear();

      frame = -1;
    }


    if(go || bStep) {
      bStep = false;
      cudaMemcpy(dKinectMeters.ptr, depthBuffer, w*h*sizeof(float), cudaMemcpyHostToDevice);
      roo::BilateralFilter<float,float>(kin_d[0],dKinectMeters,bigs,bigr,biwin,0.2);

      roo::BoxReduceIgnoreInvalid<float,MaxLevels,float>(kin_d);
      for(int l=0; l<MaxLevels; ++l) {
        roo::DepthToVbo<float>(kin_v[l], kin_d[l], K[l] );
        roo::NormalsFromVbo(kin_n[l], kin_v[l]);
      }
    }

    Sophus::SE3d T_vw(s_cam.GetModelViewMatrix());
    if(viewonly) {

      const roo::BoundingBox roi(T_vw.inverse().matrix3x4(), w, h, K, 0, 50);
      roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
      if(work_vol.IsValid()) {
        roo::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, T_vw.inverse().matrix3x4(), K, 0.1, 50, trunc_dist, true );

        if(keyframes.size() > 0) {
          // populate kfs
          for( int k=0; k< kfs.Rows(); k++)
          {
            if(k < keyframes.size()) {
              kfs[k].img = keyframes[k]->img;
              kfs[k].T_iw = keyframes[k]->T_iw.matrix3x4();
              kfs[k].K = roo::ImageIntrinsics(rgb_fl, kfs[k].img);
            }else{
              kfs[k].img.ptr = 0;
            }
          }
          roo::TextureDepth<float4,uchar3,10>(ray_c[0], kfs, ray_d[0], ray_n[0], ray_i[0], T_vw.inverse().matrix3x4(), K);
        }
      }
    }else{
      const roo::BoundingBox roi(roo::BoundingBox(T_wl.inverse().matrix3x4(), w, h, K, 0, 50));
      roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
      if(work_vol.IsValid()) {
        for(int l=0; l<MaxLevels; ++l) {
          if(its[l] > 0) {
            const roo::ImageIntrinsics Kl = K[l];
            roo::RaycastSdf(ray_d[l], ray_n[l], ray_i[l], work_vol, T_vw.inverse().matrix3x4(), Kl, knear,kfar, trunc_dist, true );
            roo::DepthToVbo<float>(ray_v[l], ray_d[l], Kl );
          }
        }

        if(fuse) {
          std::cout<<"Fusing . . . "<<std::endl;
          Sophus::SE3d tempPose = start.inverse() * last;
          const roo::BoundingBox roi(tempPose.matrix3x4(), w, h, K, knear,kfar);
          roo::BoundedVolume<roo::SDF_t> work_vol = vol.SubBoundingVolume( roi );
          //            roo::RaycastSdf(ray_d[0], ray_n[0], ray_i[0], work_vol, tempPose.matrix3x4(), K, knear, kfar, trunc_dist, true );
          if(work_vol.IsValid()) {
            const float trunc_dist = trunc_dist_factor*length(vol.VoxelSizeUnits());
            roo::SdfFuse(work_vol, kin_d[0], kin_n[0], tempPose.inverse().matrix3x4(), K, trunc_dist, max_w, mincostheta );
          }
          fuse = false;
        }
      }
    }

    glcamera.SetPose(current_pose.matrix());

    roo::BoundingBox bbox_work(current_pose.matrix3x4(), w, h, K.fu, K.fv, K.u0, K.v0, knear,kfar);
    bbox_work.Intersect(vol.bbox);
    glboxfrustum.SetBounds(roo::ToEigen(bbox_work.Min()), roo::ToEigen(bbox_work.Max()) );

    /////////////////////////////////////////////////////////////
    // Draw
    addebug.SetImage(dDebug.SubImage(0,0,w>>show_level,h>>show_level));
    adnormals.SetLevel(show_level);
    adrayimg.SetLevel(viewonly? 0 : show_level);

     depthView.SetImage();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1,1,1);
    pangolin::FinishFrame();
  }
  return 0;
}
