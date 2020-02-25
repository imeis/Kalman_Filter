#ifndef UKF_TEST_H
#define UKF_TEST_H

#include "ukf.h"
#include "Eigen/Dense"
#include "measurement_package.h"

class UKF_TEST
{
public:
	UKF_TEST();
	virtual ~UKF_TEST();
	bool CmpVec(Eigen::VectorXd v1, Eigen::VectorXd v2);
	bool CmpMat(Eigen::MatrixXd m1, Eigen::MatrixXd m2);
	void Test_NormVal();
	void Test_MeanVal();
	void Test_InitX();
	void Test_GenerateSigmaPoints();
	void Test_AugmentState();
	void Test_PredictSigmaPoints();
	void Test_SetWeights();
	void Test_CorMat();
	UKF ukf_;
	double thr_;
};
#endif 