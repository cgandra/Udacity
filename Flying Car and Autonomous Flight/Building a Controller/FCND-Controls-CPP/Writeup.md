## Project: Control of a 3D Quadrotor

## Overview  
For this project, you will modify a C++ controller to write the low level flight controllers for the vehicle to follow a timed trajectory. The goal of the trajectory following will be to arrive at the end within the specified time while maintaining position errors below a threshold

You will also be using the simulator to fly some difference trajectories to test out the performance of your C++ implementation of your controller. These trajectories, along with supporting code, are found in the `traj` directory of the repo.

## Development Environment Setup

Download or clone repository https://github.com/udacity/FCND-Controls-CPP

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
1. Implement calculating the motor commands given commanded thrust and moments.
   ```cpp
   GenerateMotorCommands() {
      float l = L/sqrt(2.f); // perpendicular distance to axes
      float c_b = collThrustCmd;
      float p_b = momentCmd.x / l;
      float q_b = momentCmd.y / l;
      float r_b = -momentCmd.z / kappa;

      cmd.desiredThrustsN[0] = CONSTRAIN((c_b + p_b + q_b + r_b) / 4.f, minMotorThrust, maxMotorThrust); // front left
      cmd.desiredThrustsN[1] = CONSTRAIN((c_b - p_b + q_b - r_b) / 4.f, minMotorThrust, maxMotorThrust); // front right
      cmd.desiredThrustsN[2] = CONSTRAIN((c_b + p_b - q_b - r_b) / 4.f, minMotorThrust, maxMotorThrust); // rear left
      cmd.desiredThrustsN[3] = CONSTRAIN((c_b - p_b - q_b + r_b) / 4.f, minMotorThrust, maxMotorThrust); // rear right
   }
   ```

2. Implement body rate control: The controller should be a proportional controller on body rates to commanded moments. The controller should take into account the moments of inertia of the drone when calculating the commanded moments. Tune `kpPQR` in `QuadControlParams.txt` to get the vehicle to stop spinning quickly but not overshoot. If successful, you should see the rotation of the vehicle about roll (omega.x) get controlled to 0 while other rates remain zero. For the simulation, use [Scenario 2](#scenario-2) for tuning/testing.

   ```cpp
   BodyRateControl() {
      V3F momentCmd;
      V3F err = pqrCmd - pqr;
      V3F I(Ixx, Iyy, Izz);
      momentCmd = I * kpPQR * err;
   }
   ```

   See [Scenario 2 result](#scenario-2)

 
3. Implement roll / pitch control: The controller should use the acceleration and thrust commands, in addition to the vehicle attitude to output a body rate command. The controller should account for the non-linear transformation from local accelerations to body rates. Note that the drone's mass should be accounted for when calculating the target angles. Tune `kpBank` in `QuadControlParams.txt` to minimize settling time but avoid too much overshoot. If successful you should now see the quad level itself. You should also see the vehicle angle (Roll) get controlled to 0. For the simulation, use [Scenario 2](#scenario-2) for tuning/testing.

   ```cpp
   RollPitchControl() {
      V3F pqrCmd(0.f, 0.f, 0.f);
      Mat3x3F R = attitude.RotationMatrix_IwrtB();

      if (collThrustCmd > 0.0) {
         float c = -collThrustCmd/mass;
	     float b_x_c = CONSTRAIN(accelCmd[0]/c, -maxTiltAngle, maxTiltAngle);
	     float b_y_c = CONSTRAIN(accelCmd[1]/c, -maxTiltAngle, maxTiltAngle);

	     float b_x_c_dot = kpBank * (b_x_c - R(0, 2));
	     float b_y_c_dot = kpBank * (b_y_c - R(1, 2));

	     pqrCmd[0] = (R(1, 0)*b_x_c_dot - R(0, 0)*b_y_c_dot) / R(2, 2);
	     pqrCmd[1] = (R(1, 1)*b_x_c_dot - R(0, 1)*b_y_c_dot) / R(2, 2);
      }
   }
   ```

   See [Scenario 2 result](#scenario-2)

4. Implement lateral position control: The controller should use the local NE position and velocity to generate a commanded local acceleration. Tune parameters `kpPosXY` and `kpVelXY` for settling time. For the simulation, use [Scenario 3](#scenario-3) for tuning/testing. If successful, the quads should be going to their destination points and tracking error should be going down

   ```cpp
   LateralPositionControl() {
      // make sure we don't have any incoming z-component
      accelCmdFF.z = 0;
      velCmd.z = 0;
      vel.z = 0;
      posCmd.z = pos.z;

      // we initialize the returned desired acceleration to the feed-forward value.
      // Make sure to _add_, not simply replace, the result of your controller
      // to this variable
      V3F accelCmd = accelCmdFF;

      V3F pTerm = kpPosXY * (posCmd - pos);
      velCmd += pTerm;

      float velMag = velCmd.magXY();
      if (velMag > maxSpeedXY) {
         velCmd = (velCmd*maxSpeedXY) / velMag;
      }

      accelCmd += kpVelXY*(velCmd - vel);
      float accelMag = accelCmd.magXY();
      if (accelMag > maxAccelXY) {
         accelCmd = (accelCmd * maxAccelXY) / accelMag;
      }
   }
   ```

   See [Scenario 3 result](#scenario-3)

5. Implement altitude controller: The controller should use both the down position and the down velocity to command thrust. It should ensure that the output value is indeed thrust (the drone's mass needs to be accounted for) and that the thrust includes the non-linear effects from non-zero roll/pitch angles. Additionally, the altitude controller should contain an integrator to handle the weight non-idealities presented in [Scenario 4](#scenario-4). Tune parameters `kpPosZ`, `kpVelZ` and `KiPosZ` for settling time. For the simulation, use [Scenario 3](#scenario-3) for tuning/testing. If successful, the quads should be going to their destination points and tracking error should be going down

   ```cpp
   AltitudeControl() {
      V3F pqrCmd(0.f, 0.f, 0.f);
      //position
      float zErr = (posZCmd - posZ);
      float pTerm = kpPosZ * zErr;

      //integral
      integratedAltitudeError += (zErr * dt);
      float iTerm = KiPosZ * integratedAltitudeError;

      //velocity
      velZCmd += pTerm;
      //Limit the ascent / descent rate
      velZCmd = CONSTRAIN(velZCmd, -maxDescentRate, maxAscentRate);
      float zErrDot = (velZCmd - velZ);
      float dTerm = kpVelZ * zErrDot;

      float u1Bar = iTerm + dTerm + accelZCmd;
      thrust = -mass * (u1Bar - 9.81f)/R(2, 2);
   }
   ```

   See [Scenario 3 result](#scenario-3)

6. Implement yaw control: The controller can be a linear/proportional heading controller to yaw rate commands (non-linear transformation not required). Tune parameters `kpYaw` and the 3rd (z) component of `kpPQR` for settling time. Don’t try to tune yaw control too tightly, as yaw control requires a lot of control authority from a quadcopter and can really affect other degrees of freedom.  This is why you often see quadcopters with tilted motors, better yaw authority!. For the simulation, use [Scenario 3](#scenario-3) for tuning/testing. If successful, the quads should be going to their destination points and tracking error should be going down

   ```cpp
   YawControl() {
      float yawRateCmd;
      float err = (yawCmd - yaw);
      err = fmodf(err, 2 * F_PI);

      yawRateCmd = kpYaw * err;
   }
   ```

   See [Scenario 3 result](#scenario-3)

7. Explore some of the non-idealities and robustness of a controller: For the simulation, use [Scenario 4](#scenario-4) for tuning/testing. If needed tweak the controller parameters to work for all 3 quads

    - Do all the quads seem to be moving OK? 'Yes'

   See [Scenario 4 result](#scenario-4)

8. Test performance for all functions: For the simulation, use [Scenario 5](#scenario-5) for tuning/testing
    - How well is your drone able to follow the trajectory?  The 'FigureEightFF.txt' yellow quad follows trajectory well, but the orange quad with 'FigureEight.txt' does not
    - Is it able to hold to the path fairly well?  'Yes' for the yellow quad

   See [Scenario 5 result](#scenario-5)
 
9. **Extra Challenge 1** of [Scenario 5](#scenario-5): Modifications made to trajectory 'traj/FigureEight.txt' to add velocity information. Changes made to script 'MakePeriodicTrajectory.py'.

   See [Scenario 5a result](#scenario-5a)

10. Test performance for all functions with [X_TestManyQuads](#x_TestManyQuads)

    See [X_TestManyQuads result](#x_TestManyQuads)

11. Ensure [Intro scenario](#scenario-1) works

    See [Intro scenario](#scenario-1)

## Flight Evaluation 

### Results
Download video files from https://www.dropbox.com/sh/rrvzk2p6sbjss06/AABw0m9qZc5-JsqshLap1TQ5a?dl=0. Copy to folder results

#### Scenario 1

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_1.mp4" type="video/mp4">
   </video>

#### Scenario 2
In this scenario, you will see a quad above the origin.  It is created with a small initial rotation speed about its roll axis.  Your controller will need to stabilize the rotational motion and bring the vehicle back to level attitude.
   
   <video width="480" height="270" controls="controls">
     <source src="results/scenario_2.mp4" type="video/mp4">
   </video>

#### Scenario 3
This will create 2 identical quads, one offset from its target point (but initialized with yaw = 0) and second offset from target point but yaw = 45 degrees.

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_3.mp4" type="video/mp4">
   </video>

#### Scenario 4
This is a configuration with 3 quads that are all are trying to move one meter forward.  However, this time, these quads are all a bit different:
 - The green quad has its center of mass shifted back
 - The orange vehicle is an ideal quad
 - The red vehicle is heavier than usual
 
   <video width="480" height="270" controls="controls">
     <source src="results/scenario_4.mp4" type="video/mp4">
   </video>

#### Scenario 5
This scenario has two quadcopters:
 - the orange one is following `traj/FigureEight.txt`
 - the other one is following `traj/FigureEightFF.txt` - for now this is the same trajectory.  For those interested in seeing how you might be able to improve the performance of your drone by adjusting how the trajectory is defined, check out **Extra Challenge 1** below!

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_5.mp4" type="video/mp4">
   </video>

#### Scenario 5a
Same as [Scenario 5](#scenario-5) with handling of **Extra Challenge 1**. Modifications made to trajectory 'traj/FigureEight.txt' to add velocity information.

   <video width="480" height="270" controls="controls">
     <source src="results/scenario_5a.mp4" type="video/mp4">
   </video>

#### X_TestManyQuads
This scenario has 9 quads that follow same trajectory but with time and x position offset

   <video width="480" height="270" controls="controls">
     <source src="results/X_TestManyQuads.mp4" type="video/mp4">
   </video>


### Performance Metrics
The specific performance metrics are as follows:
 - scenario 1
   ```
   PASS: ABS(Quad.PosFollowErr) was less than 0.500000 for at least 0.800000 seconds
   ```

 - scenario 2
   - roll should less than 0.025 radian of nominal for 0.75 seconds (3/4 of the duration of the loop)
   - roll rate should less than 2.5 radian/sec for 0.75 seconds
   ```
   PASS: ABS(Quad.Roll) was less than 0.025000 for at least 0.750000 seconds
   PASS: ABS(Quad.Omega.X) was less than 2.500000 for at least 0.750000 seconds
   ```

 - scenario 3
   - X position of both drones should be within 0.1 meters of the target for at least 1.25 seconds
   - Quad2 yaw should be within 0.1 of the target for at least 1 second
   ```
   PASS: ABS(Quad1.Pos.X) was less than 0.100000 for at least 1.250000 seconds
   PASS: ABS(Quad2.Pos.X) was less than 0.100000 for at least 1.250000 seconds
   PASS: ABS(Quad2.Yaw) was less than 0.100000 for at least 1.000000 seconds
   ```

 - scenario 4
   - position error for all 3 quads should be less than 0.1 meters for at least 1.5 seconds
   ```
   PASS: ABS(Quad1.PosFollowErr) was less than 0.100000 for at least 1.500000 seconds
   PASS: ABS(Quad2.PosFollowErr) was less than 0.100000 for at least 1.500000 seconds
   PASS: ABS(Quad3.PosFollowErr) was less than 0.100000 for at least 1.500000 seconds
   ```

 - scenario 5
   - position error of the quad should be less than 0.25 meters for at least 3 seconds
   ```
   PASS: ABS(Quad2.PosFollowErr) was less than 0.250000 for at least 3.000000 seconds
   ```
