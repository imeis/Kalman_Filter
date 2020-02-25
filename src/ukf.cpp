#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_=false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  
  // State dimension
  n_x_ = 5;
  
  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Previous timestamp 
  previous_timestamp_=0;

  // Vector for weights
  weights_ = VectorXd(2*n_aug_+1);

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Augmented mean vector
  x_aug_ = VectorXd::Zero(n_aug_);

  // Augmented state covariance
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);

  // Sigma point Augmented matrix for time k
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Sigma point estimated matrix for time k+1
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Lidar measurement noise covariance matrices
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0 ,
              0                      , std_laspy_ * std_laspy_ ;
  // Radar measurement noise covariance matrix 
  R_radar_ = MatrixXd(3, 3);
  R_radar_ <<  std_radr_ * std_radr_, 0                         , 0,
               0                    , std_radphi_ * std_radphi_ , 0,
               0                    , 0                         , std_radrd_*std_radrd_;
  
   radar_NIS_ = 0; 
   lidar_NIS_ = 0; 
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  // Initialize Measurement
  // ================================================================================
  // condition flag based on the measurement type
  bool isLidar = (meas_package.sensor_type_ == MeasurementPackage::LASER);
  bool isRadar = (meas_package.sensor_type_ == MeasurementPackage::RADAR);
  
  // condition flag based on the case that we should ignore the process
  bool dontLidar =  (isLidar && !use_laser_);
  bool dontRadar = (isRadar && !use_radar_);

  // ignore the processing of measurement
  if (dontLidar || dontRadar){return;}

  // for the first time step
  if (!is_initialized_) 
  {
    // initialize the state vector x_ from the measurement
    // if Lidar: Px and Py values
    // if Radar: Rho, Yaw, Yaw rate
    // x_ is vertical vector of size 5 
    InitX(meas_package);

    // Set the previous time step to first time stamp.
    previous_timestamp_ = meas_package.timestamp_;

    // switch the flag to true
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;
 
  // Prediction
  // ================================================================================
  // 3. Call the Kalman Filter prediction(dt) function
  Prediction(dt);

  // Update
  // ================================================================================
  // 4. Call the Kalman Filter update() function
  //      with the most recent raw measurements_
  if(isLidar)
  {  
    VectorXd z_diff = VectorXd::Zero(2);
    UpdateLidar(meas_package);
    z_diff = meas_package.raw_measurements_-z_pred_; 
    lidar_NIS_ = z_diff.transpose()*S_.inverse()*(z_diff);
  }
  else if (isRadar)
  {
    VectorXd z_diff = VectorXd::Zero(3);
    UpdateRadar(meas_package);
    z_diff = meas_package.raw_measurements_-z_pred_;   
    radar_NIS_ = z_diff.transpose()*S_.inverse()*(z_diff);
  }
  else
  {
    return;
  }
}

void UKF::Prediction(double dt) 
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // Generate Sigma Points \ Augmentation
  // ================================================================================
  AugmentState();
  GenerateSigmaPoints();

  // Predic Sigma Points
  // ================================================================================

  // predict sigma points
  PredictSigmaPoints(dt);

  // Predict Moments

  // set weights
  SetWeights();

  // predicted state mean (x_)
  MeanVal(Xsig_pred_, x_);

  // predicted state covariance matrix (P_)
  CovMat(Xsig_pred_, P_, x_,3);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Predict Measurements
  ////####################################################################

  // transform sigma points into measurement space
  Zsig_ =  MatrixXd::Zero(2, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    // measurement model
    Zsig_(0,i) = p_x;                                            // Px
    Zsig_(1,i) = p_y;                                           // Py
  }
  
  S_ = MatrixXd::Zero(2,2);
  z_pred_ = VectorXd::Zero(2);
  R_ = R_lidar_;
  VectorXd z_diff = VectorXd::Zero(2);
  UpdateMeasurement(meas_package.raw_measurements_);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Predict Measurements
  //####################################################################

  // transform sigma points into measurement space
  Zsig_ =  MatrixXd::Zero(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v_x = cos(yaw)*v;
    double v_y = sin(yaw)*v;

    // measurement model: 3X2n_aug_+1 Matrix
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                         // r
    Zsig_(1,i) = atan2(p_y,p_x);                                  // phi
    Zsig_(2,i) = (p_x*v_x + p_y*v_y) / std::max(0.001,Zsig_(0,i));   // r_dot
  }

  S_ = MatrixXd::Zero(3,3);
  z_pred_ = VectorXd::Zero(3);
  R_ = R_radar_;
  VectorXd z_diff = VectorXd::Zero(3);
  UpdateMeasurement(meas_package.raw_measurements_);
}

void UKF::UpdateMeasurement(const VectorXd &z)
{
  
  // mean predicted measurement
  MeanVal(Zsig_, z_pred_);
  //innovation covariance matrix S
  CovMat(Zsig_, S_, z_pred_,1);

  // add measurement noise covariance matrix
  S_ = S_ + R_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, z_pred_.size());

  // calculate cross correlation matrix
   CorMat(Tc);
  
  // Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  
  // residual
  VectorXd z_diff = z - z_pred_;
  
  
  // angle normalization
  NormVal(z_diff(1));
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S_*K.transpose();
}
void UKF::MeanVal(MatrixXd Xsig, VectorXd &x)
{
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  // iterate over sigma points
    x = x + weights_(i) * Xsig.col(i);
  }
}
void UKF::CovMat(MatrixXd Zsig, MatrixXd &S, VectorXd z_pred, int nrm)
{
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormVal(z_diff(nrm));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
}

void UKF::CorMat(MatrixXd &Tc)
{
  
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // 2n+1 simga points

    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    // angle normalization
    NormVal(z_diff(1));

     // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    NormVal(x_diff(3));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

  }

}
void UKF::NormVal(double &val)
{
  while(val>M_PI) val-=2.*M_PI; 
  while(val<-M_PI) val+=2.*M_PI; 
}
void UKF::SetWeights()
{
  //
  weights_(0) = lambda_/(lambda_+n_aug_);
  //
  for (int i=1; i<2*n_aug_+1; i++) 
  { // 2n+1 weights
    weights_(i) = 0.5/(n_aug_+lambda_);
  }
}
void UKF::PredictSigmaPoints(double dt)
{
  for (int i = 0; i< 2*n_aug_+1; ++i) 
  {
    // extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) 
    {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
    } 
    else 
    {
        px_p = p_x + v*dt*cos(yaw);
        py_p = p_y + v*dt*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
    py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
    v_p = v_p + nu_a*dt;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
    yawd_p = yawd_p + nu_yawdd*dt;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::AugmentState()
{
  // create augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
  
  // create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;  
 }

void UKF::GenerateSigmaPoints()
{
  // create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; ++i) 
  {
    Xsig_aug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
}

void UKF::InitX(MeasurementPackage meas_package)
{
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {  // laser measurement
    // set the state with the initial location and zero velocity
    x_ << meas_package.raw_measurements_[0], 
          meas_package.raw_measurements_[1], 
          0, 
          0, 
          0;
  }  
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
  {  // radar measurement
    // set the state with the initial radius and yaw value and zero velocity and yaw rate and 
    x_ << meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]), 
          meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]), 
          0, 
          0,
          0;        
  }
}