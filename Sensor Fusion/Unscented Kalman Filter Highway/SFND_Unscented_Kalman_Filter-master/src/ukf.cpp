#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // set state dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 4.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3.0;
  
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

  double var_laspx = std_laspx_*std_laspx_;
  double var_laspy = std_laspy_*std_laspy_;
  double var_radr = std_radr_*std_radr_;
  double var_radphi = std_radphi_*std_radphi_;
  double var_radrd = std_radrd_*std_radrd_;

  P_(0, 0) = var_laspx;
  P_(1, 1) = var_laspy;
  P_(2, 2) = var_radr;
  P_(3, 3) = var_radphi;
  P_(4, 4) = var_radrd;

  // set augmented dimension
  n_aug_ = 7;
  n_aug_sigma_ = 2*n_aug_+1;

  // set measurement dimension, radar can measure r, phi, and r_dot
  lambda_ = 3-n_aug_;

  double lmabda_n_aug = lambda_+n_aug_;
  lambda_sqrt_mul_ = sqrt(lmabda_n_aug);

  weights_ = VectorXd(n_aug_sigma_);
  double weight = 0.5/lmabda_n_aug;
  weights_.fill(weight);
  weights_(0) = lambda_/lmabda_n_aug;

  // create Sigma points matrix
  Xsig_aug_ = MatrixXd::Zero(n_aug_, n_aug_sigma_);

  // create augmented state covariance
  P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug_(n_x_, n_x_) = std_a_*std_a_;
  P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  // create predicted Sigma points matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_aug_sigma_);
  Xsig_pred_mc_ = MatrixXd::Zero(n_x_, n_aug_sigma_);
  is_initialized_ = false;

  // measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << var_laspx, 0,
	  0, var_laspy;

  // measurement noise covariance matrix
  R_radar_ = MatrixXd::Zero(3, 3);
  R_radar_(0, 0) = var_radr;
  R_radar_(1, 1) = var_radphi;
  R_radar_(2, 2) = var_radrd;

  nis_lidar_ = nis_radar_ = 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
	if (!is_initialized_) {
		InitState(meas_package);
		return;
	}

	double delta_t = (meas_package.timestamp_ - time_us_) / 1.0e6;
	time_us_ = meas_package.timestamp_;

	Prediction(delta_t);

	if (use_laser_ &&  (meas_package.sensor_type_ == MeasurementPackage::LASER)) {
		UpdateLidar(meas_package);
	}

	if (use_radar_ && (meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
		UpdateRadar(meas_package);
	}
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
	GenerateSigmaPoints();
	SigmaPointPrediction(delta_t);
	PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
	// create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd::Zero(2, n_aug_sigma_);
	Zsig.row(0) = Xsig_pred_.row(0);
	Zsig.row(1) = Xsig_pred_.row(1);

	nis_lidar_ = UpdateCommon(Zsig, R_lidar_, meas_package.raw_measurements_);
#ifdef LOG
	std::cout << "NIS_L" << "\t" << nis_lidar_ << std::endl;
#endif
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
	// create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd::Zero(3, n_aug_sigma_);

	double p_x, p_y, v, yaw;
	double d;
	for (int i = 0; i < n_aug_sigma_; i++) {
		// extract values for better readability
		p_x = Xsig_pred_(0, i);
		p_y = Xsig_pred_(1, i);
		v = Xsig_pred_(2, i);
		yaw = Xsig_pred_(3, i);

		d = sqrt(p_x*p_x + p_y*p_y);
		// measurement model
		if (fabs(d) < 0.0001) {
			d = 0.0001;
		}
		Zsig.col(i) << d, atan2(p_y, p_x), ((p_x*cos(yaw)+p_y*sin(yaw))*v)/d;
	}

	nis_radar_ = UpdateCommon(Zsig, R_radar_, meas_package.raw_measurements_);
#ifdef LOG
	std::cout << "NIS_R" << "\t" << nis_radar_ << std::endl;
#endif
}

void UKF::InitState(MeasurementPackage &meas_package) {
	/**
	 * TODO: Complete this function! Make sure you switch between lidar and radar
	 * measurements.
	 */
	 // set the state with the initial location and zero vel_abs yaw_angle yaw_rate
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		double rho = meas_package.raw_measurements_[0];
		double phi = meas_package.raw_measurements_[1];
		x_ << rho*cos(phi), rho*sin(phi), 0.0, 0.0, 0.0;
	}
	else {
		return;
	}

	time_us_ = meas_package.timestamp_;
	is_initialized_ = true;
}

void UKF::GenerateSigmaPoints() {

	// set augmented covariance matrix
	P_aug_.block(0, 0, n_x_, n_x_) = P_;

	// calculate square root of P
	MatrixXd A = P_aug_.llt().matrixL();
	A *= lambda_sqrt_mul_;

	// create sigma point matrix
	Xsig_aug_.col(0).head(n_x_) = x_;
	Xsig_aug_.block(0, 1, n_aug_, n_aug_) = A.colwise()+Xsig_aug_.col(0);
	Xsig_aug_.block(0, n_aug_ + 1, n_aug_, n_aug_) = (-1*A).colwise()+Xsig_aug_.col(0);
}

void UKF::SigmaPointPrediction(double delta_t) {
	// predict sigma points
	VectorXd noise = VectorXd(n_x_);
	VectorXd model = VectorXd(n_x_);

	double tmp;
	double yawd_delta, angle, nua_delta, nu_yawdd_delta;
	double v, yaw, yawd, nu_a, nu_yawdd;
	for (int i=0; i<n_aug_sigma_; i++) {
		// extract values for better readability
		v = Xsig_aug_(2, i);
		yaw = Xsig_aug_(3, i);
		yawd = Xsig_aug_(4, i);
		nu_a = Xsig_aug_(5, i);
		nu_yawdd = Xsig_aug_(6, i);

		yawd_delta = yawd*delta_t;
		// avoid division by zero
		if (fabs(yawd) > 0.001) {
			angle = yaw+yawd_delta;
			tmp = v/yawd;
			model << tmp*(sin(angle)-sin(yaw)), 
				tmp*(cos(yaw)-cos(angle)),
				0,
				yawd_delta,
				0;
		}
		else {
			// v*delta_t
			tmp = v*delta_t;
			model << tmp*cos(yaw),
				tmp*sin(yaw),
				0,
				yawd_delta,
				0;
		}

		// nu_a*delta_t
		nua_delta = nu_a*delta_t;
		// nu_a*delta_t*delta_t
		tmp = 0.5*nua_delta*delta_t;
		// nu_yawdd*delta_t
		nu_yawdd_delta = nu_yawdd*delta_t;

		noise << tmp*cos(yaw),
			tmp*sin(yaw),
			nua_delta,
			0.5*nu_yawdd_delta*delta_t,
			nu_yawdd_delta;

		// write predicted sigma points into right column
		Xsig_pred_.col(i) = Xsig_aug_.col(i).head(n_x_) + model + noise;
	}
}

void UKF::PredictMeanAndCovariance() {
	x_ = Xsig_pred_*weights_;

	// predict state covariance matrix
	P_.fill(0.0);
	VectorXd xdiff;
	for (int i = 0; i < n_aug_sigma_; i++) {
		Xsig_pred_mc_.col(i) = Xsig_pred_.col(i)-x_;
		NormalizeAngle(Xsig_pred_mc_(3, i));
		P_ += weights_(i)*(Xsig_pred_mc_.col(i)*Xsig_pred_mc_.col(i).transpose());
	}
}

double UKF::UpdateCommon(MatrixXd &Zsig, MatrixXd &R, VectorXd &raw_measurements) {
	// mean predicted measurement
	VectorXd z_pred = VectorXd(Zsig.rows());
	// measurement covariance matrix S
	MatrixXd S = MatrixXd::Zero(Zsig.rows(), Zsig.rows());
	// create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd::Zero(n_x_, Zsig.rows());
	PredictMeasurement(Zsig, z_pred, S, Tc, R);
	VectorXd zdiff = raw_measurements - z_pred;
	return UpdateState(zdiff, Tc, S);
}

void UKF::PredictMeasurement(MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S, MatrixXd &Tc, MatrixXd &R) {
	z_pred = Zsig*weights_;

	// calculate measurement covariance matrix S
	// Further optimization can be done by combining this function with UpdateState, computing transpose once, multiplying with weight once
	for (int i = 0; i < n_aug_sigma_; i++) {
		Zsig.col(i) -= z_pred;
		NormalizeAngle(Zsig(1, i));
		S += weights_(i)*(Zsig.col(i)*Zsig.col(i).transpose());
		// calculate cross correlation matrix
		Tc += weights_(i)*Xsig_pred_mc_.col(i)*Zsig.col(i).transpose();
	}

	S += R;
}

double UKF::UpdateState(VectorXd &zdiff, MatrixXd &Tc, MatrixXd &S) {

	// calculate Kalman gain K;
	MatrixXd SI = S.inverse();
	MatrixXd K = Tc*SI;

	// update state mean and covariance matrix
	NormalizeAngle(zdiff(1));
	x_ += (K*zdiff);
	P_ -= (K*S*K.transpose());

	// Calculate NIS, Normalized innovation squared
	double nis = zdiff.transpose()*SI*zdiff;
	return nis;
}

// angle normalization
void UKF::NormalizeAngle(double &angle) {
	while (angle > M_PI) angle -= 2.*M_PI;
	while (angle < -M_PI) angle += 2.*M_PI;
}