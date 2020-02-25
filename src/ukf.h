#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);


  /**
   * InitX initializes the state vector
   * @param meas_package the measurement of time k+1
   */
  void InitX(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  void UpdateMeasurement(const Eigen::VectorXd &z);

  void AugmentState();

  void GenerateSigmaPoints();

  void PredictSigmaPoints(double dt);

  void SetWeights();

  void MeanVal(Eigen::MatrixXd Xsig, Eigen::VectorXd &x);

  void CorMat(Eigen::MatrixXd &Tc);

  void CovMat(Eigen::MatrixXd Zsig, Eigen::MatrixXd &S, Eigen::VectorXd z_pred, int nrm);

  void NormVal(double &val);

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

   // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  //----------------------
  double previous_timestamp_;


  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // create augmented mean vector
  Eigen::VectorXd x_aug_;

  // create augmented state covariance
  Eigen::MatrixXd P_aug_;

  // create sigma point matrix
  Eigen::MatrixXd Xsig_aug_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // measurement covariance matrix S
  Eigen::MatrixXd S_;

  //measurement noise covariance matrices
  Eigen::MatrixXd R_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd R_lidar_;

  // create matrix for sigma points in measurement space
  Eigen::MatrixXd Zsig_;
  
  // mean predicted measurement
  Eigen::VectorXd z_pred_;
 
  double radar_NIS_; 
  double lidar_NIS_; 
};

#endif  // UKF_H