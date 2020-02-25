#include "ukf_test.h"
#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"
#include <assert.h>     /* assert */

using Eigen::MatrixXd;
using Eigen::VectorXd;

UKF_TEST::UKF_TEST()
{
	ukf_=UKF();
	thr_ = 0.001;
}
UKF_TEST::~UKF_TEST(){}

bool UKF_TEST::CmpVec(VectorXd v1, VectorXd v2)
{
	if (v1.size()!=v2.size())
		return false;
	for (int i=0;i<v1.size();i++)
		if (fabs(v1(i)-v2(i))>thr_)
			return false;
	return true;
}

bool UKF_TEST::CmpMat(MatrixXd m1, MatrixXd m2)
{
	if (m1.rows()!=m2.rows() || m1.cols()!=m2.cols())
		return false;
	for (int i=0;i<m1.rows();i++)
		for (int j=0;j<m1.cols();j++)
			if (fabs(m1(i,j)-m2(i,j))>thr_)
				return false;
	return true;
}

void UKF_TEST::Test_NormVal()
{
	double val=3*M_PI;
	ukf_.NormVal(val);
	assert(fabs(val-M_PI)<=thr_);
	cout<<"NormVal passed the test successfully!"<<endl;
}

void UKF_TEST::Test_MeanVal()
{
	VectorXd x = VectorXd(5);
	MatrixXd Xsig = MatrixXd(5,15);
	ukf_.weights_.fill(1.0);
	Xsig.fill(1.0);
	 
	ukf_.MeanVal(Xsig, x);
	VectorXd v = VectorXd(5); 
	v.fill(15.0);
	assert(CmpVec(x,v));
	cout<<"MeanVal passed the test successfully!"<<endl;
}

void UKF_TEST::Test_InitX()
{
	MeasurementPackage mp_lidar,mp_radar;
	VectorXd x_lidar = VectorXd(5), x_radar = VectorXd(5);
	x_lidar << 1,1,0,0,0;
	x_radar << 1*cos(2),1*sin(2),0,0,0;

	mp_lidar.sensor_type_ = MeasurementPackage::LASER;
	mp_lidar.raw_measurements_ = VectorXd(2);
	mp_lidar.raw_measurements_ << 1,1;

	mp_radar.sensor_type_ = MeasurementPackage::RADAR;
	mp_radar.raw_measurements_ = VectorXd(2);
	mp_radar.raw_measurements_ << 1,2;

	ukf_.InitX(mp_lidar);
	assert(CmpVec(ukf_.x_,x_lidar));

	ukf_.InitX(mp_radar);
	assert(CmpVec(ukf_.x_,x_radar));

	cout<<"InitX passed the test successfully!"<<endl;
}

void UKF_TEST::Test_GenerateSigmaPoints()
{
	ukf_.n_aug_ = 5;
	ukf_.lambda_ = -2;
	ukf_.x_aug_ = VectorXd(5);
	ukf_.x_aug_ <<	5.7441,
			        1.3800,
          			2.2049,
         			0.5015,
         			0.3528;
	ukf_.P_aug_ = MatrixXd(5,5);
	ukf_.P_aug_ <<   0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          			-0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           			 0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          			-0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          			-0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
	ukf_.Xsig_aug_ = MatrixXd(5,11);
	ukf_.GenerateSigmaPoints();
	MatrixXd Xsig_exp = MatrixXd(5, 11);
	Xsig_exp << 
	5.7441, 5.85768,  5.7441, 5.7441, 5.7441, 5.7441, 5.63052,  5.7441, 5.7441, 5.7441, 5.7441,
	1.38, 1.34566, 1.52806,   1.38,   1.38,   1.38, 1.41434, 1.23194,   1.38,   1.38,   1.38,
    2.2049, 2.28414, 2.24557,2.29582, 2.2049, 2.2049, 2.12566, 2.16423,2.11398, 2.2049,  2.2049,
    0.5015, 0.44339,0.631886,0.516923, 0.595227,0.5015,0.55961,0.371114,0.486077, 0.407773,0.5015,
    0.3528, 0.299973,0.462123, 0.376339, 0.48417, 0.418721, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879;
	assert(CmpMat(ukf_.Xsig_aug_,Xsig_exp));
   	cout<< "GenerateSigmaPoints passed the test successfully!"<<endl;
}

void UKF_TEST::Test_AugmentState()
{

	ukf_.x_ = VectorXd(5);
	ukf_.P_ = MatrixXd(5,5);
	ukf_.x_aug_ =VectorXd(7);
	ukf_.P_aug_ = MatrixXd(7,7);
	VectorXd x_aug_exp = VectorXd(7);
	MatrixXd P_aug_exp = MatrixXd(7,7);
	
	ukf_.std_yawdd_= 0.2;
	ukf_.std_a_ = 0.2;

	ukf_.x_ <<	5.7441,
         	 	1.3800,
          		2.2049,
          		0.5015,
          		0.3528;

	ukf_.P_ <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
		       -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
		        0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
		       -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
		       -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

	x_aug_exp  <<
				5.7441,
				  1.38,
				2.2049,
				0.5015,
				0.3528,
				     0,
				     0;		       
	P_aug_exp <<
 				0.0043,   -0.0013,   0.003, -0.0022,  -0.002,       0,       0,
			   -0.0013,    0.0077,  0.0011,  0.0071,   0.006,       0,       0,
  				0.003, 	   0.0011,  0.0054,  0.0007,  0.0008,       0,       0,
			   -0.0022,    0.0071,  0.0007,  0.0098,    0.01,       0,       0,
 			   -0.002,      0.006,  0.0008,    0.01,  0.0123,       0,       0,
           		0,              0,       0,       0,       0,    0.04,       0,
      			0,              0,       0,       0,       0,       0,    0.04;
	
	ukf_.AugmentState();
    assert(CmpVec(x_aug_exp,ukf_.x_aug_));
    assert(CmpMat(P_aug_exp,ukf_.P_aug_));

    cout<<"AugmentState passed the test successfully!"<<endl;
}

void UKF_TEST::Test_PredictSigmaPoints()
{
	double dt = 0.1;
	ukf_.n_x_ = 5;
	ukf_.n_aug_ = 7;
	ukf_.Xsig_aug_ = MatrixXd(7,15);
	ukf_.Xsig_pred_= MatrixXd(5,15);
	MatrixXd Xsig_pred_exp = MatrixXd(5,15);
	ukf_.Xsig_aug_ <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

	Xsig_pred_exp << 
	5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
	1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
	2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
	0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
	0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;
	ukf_.PredictSigmaPoints(dt);
	assert(CmpMat(Xsig_pred_exp, ukf_.Xsig_pred_));
	cout<<"PredictSigmaPoints passed the test successfully!"<<endl;	
}

void UKF_TEST::Test_SetWeights()
{
	ukf_.n_aug_ = 7;
	ukf_.lambda_ = 3 - ukf_.n_aug_;
	ukf_.weights_= VectorXd(2*ukf_.n_aug_+1);

	VectorXd weights_exp = VectorXd(2*ukf_.n_aug_+1);
	weights_exp << 
	-1.33333,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667,
	0.166667;

	ukf_.SetWeights();
	assert(CmpVec(weights_exp, ukf_.weights_));
	cout<<"SetWeights passed the test successfully!"<<endl;	
}

void UKF_TEST::Test_CorMat()
{
	
}