#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

//#define LOG
//#define LOG_MEAS

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
  void ProcessLaserMeasurement(MeasurementPackage meas_package);
  void ProcessRadarMeasurement(MeasurementPackage meas_package);

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

  // Initialize state vector
  void InitState(MeasurementPackage &meas_package);

  // Prediction stage functions
  void GenerateSigmaPoints();
  void SigmaPointPrediction(double delta_t);
  void PredictMeanAndCovariance();

  // Update stage functions
  double UpdateCommon(Eigen::MatrixXd &Zsig, Eigen::MatrixXd &R, Eigen::VectorXd &raw_measurements);
  void PredictMeasurement(Eigen::MatrixXd &Zsig, Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::MatrixXd &Tc, Eigen::MatrixXd &R);
  double UpdateState(Eigen::VectorXd &zdiff, Eigen::MatrixXd &Tc, Eigen::MatrixXd &S);
  void NormalizeAngle(double &angle);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // predicted sigma points matrix-mean
  Eigen::MatrixXd Xsig_pred_mc_;

  // time when the state is true, in us
  long long time_us_;

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

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;
  int n_aug_sigma_;

  // Sigma point spreading parameter
  double lambda_;
  double lambda_sqrt_mul_;

  // Sigma points matrix
  Eigen::MatrixXd Xsig_aug_;

  // Augmented state covariance
  Eigen::MatrixXd P_aug_;

  // measurement noise covariance matrix
  Eigen::MatrixXd R_lidar_;
  Eigen::MatrixXd R_radar_;

  // Normalized innovation squared
  double nis_lidar_, nis_radar_;
};

#endif  // UKF_H