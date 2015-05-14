#include <node/Node.h>
#include <sophus/se3.hpp>
#include <calibu/Calibu.h>
#include <unistd.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/Utils/GetPot>
#include <HAL/Messages/Pose.h>
#include <HAL/Messages/Matrix.h>
#include <HAL/NodeCamMessage.pb.h>

Eigen::Matrix3d K;
int w, h;

bool send_data_(node::node* n, hal::Camera *cam) {
	std::shared_ptr<hal::ImageArray> pImages = hal::ImageArray::Create();

	cam->Capture(*pImages);

	hal::Msg msg;
	msg.mutable_camera()->Swap(&pImages->Ref());

	return n->publish("DepthCam", msg);
}

void register_cam_(RegisterNodeCamReqMsg& req, RegisterNodeCamRepMsg& rep, void* UserData) {
	rep.set_regsiter_flag(1);
	rep.set_width(w);
	rep.set_height(h);
	hal::MatrixMsg Kmsg;
	hal::WriteMatrix(K, &Kmsg);
}

int main(int argc, char* argv[]) {
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

	K << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	if (cl.search("-cmod")) {
		std::shared_ptr<calibu::Rigd> rig = calibu::ReadXmlRig(cl.follow("cameras.xml", "-cmod"));
		K = rig->cameras_[0]->K();
	}

	w = cam->Width();
	h = cam->Height();

	client.set_verbosity(9);
	client.init(cl.follow("localsim", "-name"));
	client.advertise("DepthCam");
	client.provide_rpc("register_cam_", &register_cam_, NULL);

	while (1) {
		send_data_(&client, cam);
		usleep(1e6 / 60);
	}

	return 0;
}
