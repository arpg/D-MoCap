#include <Node/Node.h>
#include <Sophus/se3.hpp>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/Utils/GetPot>
#include <PbMsgs/DensePose.pb.h>

bool send_data_( node::node* n, hal::Camera *cam )
{
  std::shared_ptr<pb::ImageArray> pImages = pb::ImageArray::Create();
  pb::DensePoseMsg dpm;
  cam->Capture( *pImages );

//  pb::ImageMsg = dpm.image().add_image();

//  image_msg_.Swap(  );
//  n->publish("DepthImage", cam_msg_);
}

void register_cam_(pb::RegisterNodeCamReqMsg& req, pb::RegisterNodeCamRepMsg& rep, void* UserData)
{
  rep.set_regsiter_flag(1);
//  rep.set_width(width);
//  rep.set_height(height);
//  rep.set_channels(channels);
}


int main( int argc, char* argv[] )
{
  node::node client;
  GetPot cl(argc, argv);
  client.set_verbosity(9);
  client.init( cl.follow("localsim", "-name"));
  client.advertise("DepthCam");
  client.provide_rpc("register_cam_", &register_cam_, NULL);

  hal::Camera* cam;
  if (!cl.search("-cam")) {
    fprintf(stderr, "Camera must be provided.\n");
    fflush(stderr);
    exit(1);
  } else {
    cam = new hal::Camera(cl.follow("", "-cam"));
  }
  while (1) {
    send_data_( &client, cam);
  }

  return 0;
}
