/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

//#include "render/render.h"
#include "highway.h"
#include "ukf_test.h"

int main(int argc, char** argv)
{
	bool testUKF = false;

	if(!testUKF)
	{
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);

		// set camera position and angle
		viewer->initCameraParameters();
		float x_pos = 0;
		viewer->setCameraPosition ( x_pos-26, 0, 15.0, x_pos+25, 0, 0, 0, 0, 1);

		Highway highway(viewer);

		//initHighway(viewer);

		int frame_per_sec = 30;
		int sec_interval = 10;
		int frame_count = 0;
		int time_us = 0;

		double egoVelocity = 25;

		while (frame_count < (frame_per_sec*sec_interval))
		{
			viewer->removeAllPointClouds();
			viewer->removeAllShapes();

			//stepHighway(egoVelocity,time_us, frame_per_sec, viewer);
			highway.stepHighway(egoVelocity,time_us, frame_per_sec, viewer);
			viewer->spinOnce(1000/frame_per_sec);
			frame_count++;
			time_us = 1000000*frame_count/frame_per_sec;

		}
	}
	else
	{
		UKF_TEST ut;
		ut.Test_NormVal();
		ut.Test_MeanVal();
		ut.Test_InitX();
		ut.Test_GenerateSigmaPoints();
		ut.Test_AugmentState();
		ut.Test_PredictSigmaPoints();
		ut.Test_SetWeights();
		ut.Test_CorMat();
	}

}
