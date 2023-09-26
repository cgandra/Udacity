## Project: Building an Estimator

## Overview
For this project, you will develop an estimator to be used by your controller to successfully fly a desired flight path using realistic sensors. This project is built on the same simulator you should now be familiar with from the Controls C++ project

You will also be using the simulator to fly different trajectories to test out the performance of your C++ implementation of your estimator. These trajectories, along with supporting code, are found in the `traj` directory of the repo.

## Development Environment Setup

Download or clone repository https://github.com/udacity/FCND-Estimation-CPP

Once you have the code for the simulator, you will need to install the necessary compiler and IDE necessary for running the simulator.

Here are the setup and install instructions for each of the recommended IDEs for each different OS options:

### Windows

For Windows, the recommended IDE is Visual Studio.  Here are the steps required for getting the project up and running using Visual Studio.

1. Download and install [Visual Studio](https://www.visualstudio.com/vs/community/)
2. Select *Open Project / Solution* and open `<simulator>/project/Simulator.sln`
3. From the *Project* menu, select the *Retarget solution* option and select the Windows SDK that is installed on your computer (this should have been installed when installing Visual Studio or upon opening of the project).
4. Make sure platform matches the flavor of Windows you are using (x86 or x64). The platform is visible next to the green play button in the Visual Studio toolbar:

![x64](x64.png)

5. To compile and run the project / simulator, simply click on the green play button at the top of the screen.  When you run the simulator, you should see a single quadcopter, falling down.

### The Simulator

In the simulator window itself, you can right click the window to select between a set of different scenarios that are designed to test the different parts of your controller.

The simulation (including visualization) is implemented in a single thread.  This is so that you can safely breakpoint code at any point and debug, without affecting any part of the simulation.

Due to deterministic timing and careful control over how the pseudo-random number generators are initialized and used, the simulation should be exactly repeatable. This means that any simulation with the same configuration should be exactly identical when run repeatedly or on different machines.

Vehicles are created and graphs are reset whenever a scenario is loaded. When a scenario is reset (due to an end condition such as time or user pressing the ‘R’ key), the config files are all re-read and state of the simulation/vehicles/graphs is reset -- however the number/name of vehicles and displayed graphs are left untouched.

When the simulation is running, you can use the arrow keys on your keyboard to impact forces on your drone to see how your controller reacts to outside forces being applied.

#### Keyboard / Mouse Controls

There are a handful of keyboard / mouse commands to help with the simulator itself, including applying external forces on your drone to see how your controllers reacts!

 - Left drag - rotate
 - X + left drag - pan
 - Z + left drag - zoom
 - arrow keys - apply external force
 - C - clear all graphs
 - R - reset simulation
 - Space - pause simulation

## Project Description
1. Determine the standard deviation of the measurement noise of both GPS X data and Accelerometer X data. Collect some simulated noisy sensor data and estimate the standard deviation of the quad's sensor. Use [Scenario 6](#scenario-6)

   ```python
    gpsX = np.loadtxt('results/log_mean_std/Graph1.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]
    imuAx = np.loadtxt('results/log_mean_std/Graph2.txt',delimiter=',',dtype='Float64',skiprows=1)[:,1]

    gpsX_std = np.std(gpsX)
    imuAx_std = np.std(imuAx)
    print('GPSPosXY_Std - {} with {} samples'.format(gpsX_std, len(gpsX)))
    print('AccelXY_Std - {} with {} samples'.format(imuAx_std, len(imuAx)))
   ```

   See [Scenario 6 result](#scenario-6)

   ```python
    GPSPosXY_Std - 0.7107616392538426 with 95 samples
    AccelXY_Std - 0.4917373264517782 with 1915 samples
   ```

2. Implement a better rate gyro attitude integration scheme in the UpdateFromIMU() function. The improved integration scheme should result in an attitude estimator of < 0.1 rad for each of the Euler angles for a duration of at least 3 seconds during the simulation. The integration scheme should use quaternions to improve performance over the current simple integration scheme. Use [Scenario 7](#scenario-7)

   ```cpp
   UpdateFromIMU() {
	  bool bRotMat = false;
	  float predictedPitch, predictedRoll;
	  if (bRotMat) {
	    float phi = rollEst;
	    float theta = pitchEst;
	    float v[9] = { 1.f, sin(phi)*tan(theta), cos(phi)*tan(theta), 0.f, cos(phi), -sin(phi), 0.f, sin(phi)/cos(theta), cos(phi)/cos(theta) };
	    Mat3x3F eulerRot = Mat3x3F(v);
	    V3F eulerDot = eulerRot * gyro;

  	    predictedPitch = pitchEst + dtIMU * eulerDot.y;
	    predictedRoll = rollEst + dtIMU * eulerDot.x;
	    ekfState(6) +=  dtIMU * eulerDot.z; // yaw
	  }
	  else {
	    Quaternion<float> qt = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, ekfState(6));
	    qt = qt.IntegrateBodyRate(gyro, dtIMU);

	    predictedPitch = qt.Pitch();
	    predictedRoll = qt.Roll();
	    ekfState(6) = qt.Yaw();
	  }
   }
   ```

   See [Scenario 7 result](#scenario-7)

 
3. Implement all of the elements of the prediction step for the estimator. The prediction step should include the state update element (PredictState() function), a correct calculation of the Rgb prime matrix, and a proper update of the state covariance. The acceleration should be accounted for as a command in the calculation of gPrime. The covariance update should follow the classic EKF update equation.

   * Implement the state prediction step. For the simulation, use [Scenario 8](#scenario-8) with perfect IMU for testing.
      ```cpp
      PredictState() {
	    Quaternion<float> attitude = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, curState(6));

	    predictedState(0) = curState(0) + curState(3) * dt;
	    predictedState(1) = curState(1) + curState(4) * dt;
	    predictedState(2) = curState(2) + curState(5) * dt;

	    V3F accW = attitude.Rotate_BtoI(accel);
	    predictedState(3) = curState(3) + accW.x * dt;
	    predictedState(4) = curState(4) + accW.y * dt;
	    predictedState(5) = curState(5) + (accW.z - CONST_GRAVITY) * dt;
     }
     ```

     See [Scenario 8 result](#scenario-8)

   * Calculate the partial derivative of the body-to-global rotation matrix and implement the rest of the prediction step. For the simulation, use [Scenario 9](#scenario-9) with realistic IMU with noise for tuning/testing. Tune the `QPosXYStd` and the `QVelXYStd` process parameters in `QuadEstimatorEKF.txt` to try to capture the magnitude of the error you see
      ```cpp
      GetRbgPrime() {
	    float cTheta = cos(pitch);
	    float sTheta = sin(pitch);

	    float cPhi = cos(roll);
	    float sPhi = sin(roll);

	    float cPsi = cos(yaw);
	    float sPsi = sin(yaw);

	    RbgPrime(0, 0) = -cTheta*sPsi;
	    RbgPrime(0, 1) = -sPhi*sTheta*sPsi - cPhi*cPsi;
	    RbgPrime(0, 2) = -cPhi*sTheta*sPsi + sPhi*cPsi;

	    RbgPrime(1, 0) = cTheta*cPsi;
	    RbgPrime(1, 1) = sPhi*sTheta*cPsi - cPhi*sPsi;
	    RbgPrime(1, 2) = cPhi*sTheta*cPsi + sPhi*sPsi;
      }

      Predict() {
	    // we'll want the partial derivative of the Rbg matrix
	    MatrixXf RbgPrime = GetRbgPrime(rollEst, pitchEst, ekfState(6));

	    // we've created an empty Jacobian for you, currently simply set to identity
	    MatrixXf gPrime(QUAD_EKF_NUM_STATES, QUAD_EKF_NUM_STATES);
	    gPrime.setIdentity();

	    VectorXf u(3);
	    u << accel[0], accel[1], accel[2];
	    VectorXf RbgPrimeU = RbgPrime*u;

	    gPrime(0, 3) = dt;
	    gPrime(1, 4) = dt;
	    gPrime(2, 5) = dt;

	    gPrime(3, 6) = RbgPrimeU(0)*dt;
	    gPrime(4, 6) = RbgPrimeU(1)*dt;
	    gPrime(5, 6) = RbgPrimeU(2)*dt;
      }
     ```

     See [Scenario 9 result](#scenario-9)

4. Implement the magnetometer update. The update should properly include the magnetometer data into the state. Note that the solution should make sure to correctly measure the angle error between the current state and the magnetometer value (error should be the short way around, not the long way). For the simulation, use [Scenario 10](#scenario-10) for tuning/testing. Tune the parameter `QYawStd` (`QuadEstimatorEKF.txt`) for the QuadEstimatorEKF so that it approximately captures the magnitude of the drift. Your goal is to both have an estimated standard deviation that accurately captures the error and maintain an error of less than 0.1rad in heading for at least 10 seconds of the simulation

   ```cpp
   UpdateFromMag() {
      VectorXf z(1), zFromX(1);
      z(0) = magYaw;

      MatrixXf hPrime(1, QUAD_EKF_NUM_STATES);
      hPrime.setZero();

      hPrime(0, 6) = 1.f;	//sec 7.3.2, 58
      float diff = ekfState(6)-magYaw;
      zFromX(0) = ekfState(6);
      if (diff > F_PI) {
	    zFromX(0) -= 2.f * F_PI;
      }
      else if (diff < -F_PI) {
	    zFromX(0) += 2.f * F_PI;
      }
      Update(z, hPrime, R_Mag, zFromX);
   }
   ```

   See [Scenario 10 result](#scenario-10)

5. Implement the GPS update. The estimator should correctly incorporate the GPS information to update the current state estimate. For the simulation, use [Scenario 11](#scenario-11) for tuning/testing. Tune the process noise model in `QuadEstimatorEKF.txt` to try to approximately capture the error you see with the estimated uncertainty (standard deviation) of the filter. Your objective is to complete the entire simulation cycle with estimated position error of < 1m (you’ll see a green box over the bottom graph if you succeed).

   ```cpp
   UpdateFromGPS() {
      VectorXf z(6), zFromX(6);
      z(0) = pos.x;
      z(1) = pos.y;
      z(2) = pos.z;
      z(3) = vel.x;
      z(4) = vel.y;
      z(5) = vel.z;

      MatrixXf hPrime(6, QUAD_EKF_NUM_STATES);
      hPrime.setZero();

      for (int i = 0; i < hPrime.rows(); i++) {
	    hPrime(i, i) = 1.f;
      }

      for (int i = 0; i < zFromX.rows(); i++) {
	    zFromX(i) = ekfState(i);
      }

      Update(z, hPrime, R_GPS, zFromX);
   }
   ```

   See [Scenario 11 result](#scenario-11)

6. Adding Your Controller from previous project. De-tune your controller to successfully fly the final desired box trajectory with your estimator and realistic sensors. Successfully meet the performance criteria of the final scenario (<1m error for entire box flight).

   See [Scenario 11a result](#scenario-11a)

7. Meet the performance criteria of each step.
   * Scenario 6. Your standard deviations should accurately capture the value of approximately 68% of the respective measurements
     ```
     PASS: ABS(Quad.GPS.X-Quad.Pos.X) was less than MeasuredStdDev_GPSPosXY for 71% of the time
     PASS: ABS(Quad.IMU.AX-0.000000) was less than MeasuredStdDev_AccelXY for 68% of the time
     ```

   * Scenario 7
     ```
     PASS: ABS(Quad.Est.E.MaxEuler) was less than 0.100000 for at least 3.000000 seconds
     ```

   * Scenario 10
     ```
     PASS: ABS(Quad.Est.E.Yaw) was less than 0.120000 for at least 10.000000 seconds
     PASS: ABS(Quad.Est.E.Yaw-0.000000) was less than Quad.Est.S.Yaw for 70% of the time
     ```

   * Scenario 11
     ```
     PASS: ABS(Quad.Est.E.Pos) was less than 1.000000 for at least 20.000000 seconds
     ```

   * Scenario 11 with previous project controller/config
     ```
     PASS: ABS(Quad.Est.E.Pos) was less than 1.000000 for at least 20.000000 seconds
     ```

## Scenario Description and Results

### Scenario 6
In this simulation, the interest is to record some sensor data on a static quad, so you will not see the quad move.  You will see two plots at the bottom, one for GPS X position and one for The accelerometer's x measurement.  The dashed lines are a visualization of a single standard deviation from 0 for each signal. The standard deviations are initially set to arbitrary values (after processing the data in the next step, you will be adjusting these values).  If they were set correctly, we should see ~68% of the measurement points fall into the +/- 1 sigma bound.  When you run this scenario, the graphs you see will be recorded to the following csv files with headers: `config/log/Graph1.txt` (GPS X data) and `config/log/Graph2.txt` (Accelerometer X data).

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_6.mp4" type="video/mp4">
   </video>

### Scenario 7
For this simulation, the only sensor used is the IMU and noise levels are set to 0 (see `config/07_AttitudeEstimation.txt` for all the settings for this simulation)   

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_7.mp4" type="video/mp4">
   </video>

### Scenario 8
This scenario is configured to use a perfect IMU (only an IMU). Due to the sensitivity of double-integration to attitude errors, we've made the accelerometer update very insignificant (`QuadEstimatorEKF.attitudeTau = 100`). The plots on this simulation show element of your estimated state and that of the true state.  At the moment you should see that your estimated state does not follow the true state

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_8.mp4" type="video/mp4">
   </video>

### Scenario 9
This scenario inlcudes a realistic IMU, one with noise. You will see a small fleet of quadcopter all using your prediction code to integrate forward. You will see two plots:
   - The top graph shows 10 (prediction-only) position X estimates
   - The bottom graph shows 10 (prediction-only) velocity estimates
 
   <video width="480" height="270" controls="controls">
     <source src="results/scenario_9.mp4" type="video/mp4">
   </video>

### Scenario 10
This scenario uses a realistic IMU, but the magnetometer update hasn’t been implemented yet. As a result, you will notice that the estimate yaw is drifting away from the real value (and the estimated standard deviation is also increasing).  Note that in this case the plot is showing you the estimated yaw error (`quad.est.e.yaw`), which is drifting away from zero as the simulation runs.  You should also see the estimated standard deviation of that state (white boundary) is also increasing.

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_10.mp4" type="video/mp4">
   </video>

### Scenario 11
This scenario is using both an ideal estimator/IMU and realistic estimator/IMU. 

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_11.mp4" type="video/mp4">
   </video>

### Scenario 11a
This scenario uses a realistic estimator/IMU. 

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_11a.mp4" type="video/mp4">
   </video>