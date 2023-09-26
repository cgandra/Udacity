# Go Chase It
Project 2 of Udacity Robotics Software Engineer Nanodegree Program

https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/Go%20Chase%20It/images/submit_15fps.mp4

<video src="https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/Go%20Chase%20It/images/submit_15fps.mp4" controls="controls" style="max-width: 480px;">
</video>


## Overview
In this project, you should create two ROS packages inside your catkin_ws/src: 
* the drive_bot
* the ball_chaser 

Here are the steps to design the robot, house it inside your world, and program it to chase white-colored balls:
  
1. **drive_bot**:
   * Create a **my_robot** ROS package to hold your robot, the white ball, and the world.
   * Design a differential drive robot with the Unified Robot Description Format. Add two sensors to your robot: a lidar and a camera. Add Gazebo plugins for your robot’s differential drive, lidar, and camera. The robot you design should be significantly different from the one presented in the project lesson. Implement significant changes such as adjusting the color, wheel radius, and chassis dimensions. Or completely redesign the robot model!
   * House your robot inside the world you built in the **Build My World** project.
   * Add a white-colored ball to your Gazebo world and save a new copy of this world.
   * The **world.launch** file should launch your world with the white-colored ball and your robot.

2. **ball_chaser**: 
   * Create a **ball_chaser** ROS package to hold your C++ nodes.
   * Write a **drive_bot** C++ node that will provide a **ball_chaser/command_robot** service to drive the robot by controlling its linear x and angular z velocities. The service should publish to the wheel joints and return back the requested velocities.
   * Write a **process_image** C++ node that reads your robot’s camera image, analyzes it to determine the presence and position of a white ball. If a white ball exists in the image, your node should request a service via a client to drive the robot towards it.
   * The **ball_chaser.launch** should run both the **drive_bot** and the **process_image** nodes.
   
## Setup Instructions
1. First, enable the GPU on your workspace by clicking Enable GPU.
2. Open the visual desktop by clicking on Go to Desktop. The workspace is best supported on **Google Chrome** and might not load on other browsers.
3. Then, update and upgrade the Workspace image to get the latest features of Gazebo. To do so, open a terminal, and write the following statement:

    ``` sudo apt-get update && sudo apt-get upgrade -y ```

    **Note**: Remember to update and upgrade your image after each reboot since these updates(or any package that you install) are not permanent. Ignore any error you get while upgrading
  
4. Build and run your code.  

## Submission Description  
Directory Structure  
```
GoChaseIt                                      # Go Chase It Project
├── setupRobot.sh                              # Shell script to setup catkin_ws
├── files                                      # All the project files used by setupRobot.sh to setup catkin_ws
│   │   │   ├── model.config
│   ├── model                                  # 3rd party and my_ball model files
│   │   ├── 3DGEMS
│   │   ├── my_ball
│   │   │   ├── model.sdf
│   │   │   ├── model.config
│   ├── ball_chaser.launch
│   ├── CMakeLists_bc.txt
│   ├── drive_bot.cpp
│   ├── DriveToTarget.srv
│   ├── empty.world
│   ├── hokuyo.dae
│   ├── my_robot.gazebo
│   ├── my_robot.xacro
│   ├── myWorld.world
│   ├── process_image.cpp
│   ├── robot_description.launch
│   ├── world.launch
├── images                                     # Video capture of result
│   ├── submit.mp4
├── launchRobot.sh                             # Shell script to launch my_robot in Gazebo to load both the world and plugins
├── launchBallChaser.sh                        # Shell script to launch ball_chaser and process_image nodes
├── visualize.sh                               # Shell script to visualize the robot’s camera images
```


## Setup the project  
* Unzip the submission
* On terminal
``` 
  cd GoChaseIt
  chmod +x *.sh
  source setupRobot.sh
```

## Project Description  
Directory Structure  

```
GoChaseIt                                      # Go Chase It Project
├── catkin_ws                                  # Catkin workspace
│   ├── src
│   │   ├── ball_chaser                        # ball_chaser package        
│   │   │   ├── launch                         # launch folder for launch files
│   │   │   │   ├── ball_chaser.launch
│   │   │   ├── src                            # source folder for C++ scripts
│   │   │   │   ├── drive_bot.cpp
│   │   │   │   ├── process_images.cpp
│   │   │   ├── srv                            # service folder for ROS services
│   │   │   │   ├── DriveToTarget.srv
│   │   │   ├── CMakeLists.txt                 # compiler instructions
│   │   │   ├── package.xml                    # package info
│   │   ├── my_robot                           # my_robot package        
│   │   │   ├── launch                         # launch folder for launch files   
│   │   │   │   ├── robot_description.launch
│   │   │   │   ├── world.launch
│   │   │   ├── meshes                         # meshes folder for sensors
│   │   │   │   ├── hokuyo.dae
│   │   │   ├── urdf                           # urdf folder for xarco files
│   │   │   │   ├── my_robot.gazebo
│   │   │   │   ├── my_robot.xacro
│   │   │   ├── worlds                         # world folder for world files
│   │   │   │   ├── empty.world
│   │   │   │   ├── office.world
│   │   │   ├── model                         # 3rd party and my_ball model files
│   │   │   │   ├── 3DGEMS
│   │   │   │   ├── my_ball
│   │   │   │   │   ├── model.config
│   │   │   │   │   ├── model.sdf
│   │   │   ├── CMakeLists.txt                 # compiler instructions
│   │   │   ├── package.xml                    # package info
├── images                                     # Video capture of result
│   ├── submit.mp4
├── launchRobot.sh                             # Shell script to launch my_robot in Gazebo to load both the world and plugins
├── launchBallChaser.sh                        # Shell script to launch ball_chaser and process_image nodes
├── visualize.sh                               # Shell script to visualize the robot’s camera images
```


- [submit.mp4](/images/submit.mp4): Video of robot chasing ball.  
- [robot_description.launch](/catkin_ws/src/my_robot/launch/robot_description.launch): Create robot model in Gazebo world. 
- [world.launch](/catkin_ws/src/my_robot/launch/world.launch): Launch myWorld model (which includes a white ball) and robot.
- [hokuyo.dae](/catkin_ws/src/my_robot/meshes/hokuyo.dae): Hokuyo LiDAR sensor mesh model.  
- [myWorld.world](/catkin_ws/src/my_robot/worlds/myWorld.world): Gazebo world file that includes the models.  
- [my_robot.gazebo](/catkin_ws/src/my_robot/urdf/my_robot.gazebo): Define my_robot URDF model plugins.  
- [my_robot.xacro](/catkin_ws/src/my_robot/urdf/my_robot.xacro): Define my_robot URDF model.
- [CMakeLists.txt](/catkin_ws/src/my_robot/CMakeLists.txt): File to link the C++ code to libraries.  
- [ball_chaser.launch](/catkin_ws/src/ball_chaser/launch/ball_chaser.launch): Run the drive_bot & process_image C++ node  
- [drive_bot.cpp](/catkin_ws/src/ball_chaser/src/drive_bot.cpp): ROS service C++ script, a ball_chaser/command_robot service. Service accepts linear x and angular z velocities, publishes to the the wheel joints and eturns the requested velocities 
- [process_images.cpp](/catkin_ws/src/ball_chaser/src/process_images.cpp): ROS service C++ script. Subscribes to the robot’s camera image. A function to analyze the image and determine the presence and position of a white ball. Requests a service to drive the robot towards a white ball (when present)
- [CMakeLists.txt](/catkin_ws/src/ball_chaser/CMakeLists.txt): File to link the C++ code to libraries.  

  
## Run the project  
```
  cd catkin_ws
  catkin_make
  cd ..
```
* Launch my_robot in Gazebo to load both the world and plugins  
```
source launchRobot.sh
```
* Launch ball_chaser and process_image nodes  
```
source launchBallChaser.sh
```
* Visualize the robot’s camera images 
```
source visualize.sh
```
