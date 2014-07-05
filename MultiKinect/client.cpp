#include <Node/Node.h>
#include <Sophus/se3.hpp>
#include <calibu/Calibu.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/Utils/GetPot>
#include <PbMsgs/DensePose.pb.h>
#include <PbMsgs/Matrix.h>

Eigen::Matrix3d K;
int w, h;

bool send_data_( node::node* n, hal::Camera *cam )
{
  std::shared_ptr<pb::ImageArray> pImages = pb::ImageArray::Create();
  pb::DensePoseMsg dpm;
  pb::CameraMsg cam_msg_ = dpm.image();
  pb::ImageMsg* pbim = cam_msg_.add_image();

  cam->Capture( *pImages );

  pbim->set_width( w );
  pbim->set_height( h );
  pbim->set_format( pb::PB_LUMINANCE );
  pbim->set_type( pb::PB_FLOAT );
  pbim->set_data( (const char*) pImages->at(0)->data() );

  return n->publish("DepthImage", dpm);
}

void register_cam_(pb::RegisterNodeCamReqMsg& req, pb::RegisterNodeCamRepMsg& rep, void* UserData)
{
  rep.set_regsiter_flag(1);
  rep.set_width(w);
  rep.set_height(h);
  pb::MatrixMsg Kmsg;
  pb::WriteMatrix( K, &Kmsg );
}


int main( int argc, char* argv[] )
{
  node::node client;
  GetPot cl(argc, argv);

  hal::Camera* cam;
  if (!cl.search("-cam")) {
    fprintf(stderr, "Camera must be provided.\n");
    fflush(stderr);
    exit(1);
  } else {
    cam = new hal::Camera(cl.follow("", "-cam"));
  }

  K << 1, 0, 0,
       0, 1, 0,
       0, 0, 1;
  if (cl.search("-cmod")) {
  calibu::CameraRig rig = calibu::ReadXmlRig(cl.follow("cameras.xml", "-cmod"));
  K = rig.cameras[0].camera.K();
  }

  w = cam->Width();
  h = cam->Height();

  client.set_verbosity(9);
  client.init( cl.follow("localsim", "-name"));
  client.advertise("DepthCam");
  client.provide_rpc("register_cam_", &register_cam_, NULL);


  while (1) {
    send_data_( &client, cam);
  }

  return 0;
}
