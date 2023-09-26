[//]: # (Image References)

[image1]: images/Capture_1.png
[image2]: images/Capture_2.png
[image3]: images/Capture_3.png
[image4]: images/Capture_4.png
[image5]: images/Capture_5.png
[image6]: images/Capture_6.png
[image7]: images/Capture_7.png
[image8]: images/ML_1.png
[image9]: images/ML_2.png
[image10]: images/P3D_1.png
[image11]: images/P3D_2.png
[image12]: images/P3D_3.png

# Map My World
Project 4 of Udacity Robotics Software Engineer Nanodegree Program

<video width="480" height="320" controls="controls">
  <source src="images/ResultTrim_git_12fps.mp4" type="video/mp4">
</video>

## Overview  
In this project you will create a 2D occupancy grid and 3D octomap from a simulated environment using your own robot with the RTAB-Map package

1. You will develop your own package to interface with the rtabmap_ros package.
2. You will build upon your localization project to make the necessary changes to interface the robot with RTAB-Map. An example of this is the addition of an RGB-D camera.
3. You will ensure that all files are in the appropriate places, all links are properly connected, naming is properly setup and topics are correctly mapped. Furthermore you will need to generate the appropriate launch files to launch the robot and map its surrounding environment.
4. When your robot is launched you will teleop around the room to generate a proper map of the environment.

## Setup Instructions
1. First, enable the GPU on your workspace by clicking Enable GPU.
2. Open the visual desktop by clicking on Go to Desktop. The workspace is best supported on **Google Chrome** and might not load on other browsers.
3. Then, update and upgrade the Workspace image to get the latest features of Gazebo
    ``` sudo apt-get update && sudo apt-get upgrade -y ```

    **Note**: Remember to update and upgrade your image after each reboot since these updates(or any package that you install) are not permanent. Ignore any error you get while upgrading

**Local VM setup**
1. Setup VM per Udacity instructions
2. Then, update and upgrade the Workspace image to get the latest features of Gazebo. To do so, open a terminal, and write the following statement:

    ``` sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116    ``` 

    ``` sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654    ``` 

    ``` wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -    ``` 

    ``` sudo apt-get update && sudo apt-get upgrade -y ```

3. Install Dependencies

    ``` sudo apt-get install --only-upgrade ros-kinetic*    ``` 

    ``` sudo apt-get install ros-kinetic-rtabmap-ros ``` 

4. Build and run your code.  

## Project Description  
Directory Structure  
```
.MapMyWorld                                    # Project
├── catkin_ws                                  # Catkin workspace
│   ├── src
│   │   ├── my_robot                           # my_robot package        
│   │   │   ├── config                         # config folder for robot navigation parameters preset 
│   │   │   │   ├── base_local_planner_params.yaml
│   │   │   │   ├── costmap_common_params.yaml
│   │   │   │   ├── global_costmap_params.yaml
│   │   │   │   ├── local_costmap_params.yaml
│   │   │   ├── database                       # folder to save map db file 
│   │   │   │   ├── rtabmap.db
│   │   │   │   ├── mesh.ply
│   │   │   ├── launch                         # launch folder for launch files   
│   │   │   │   ├── localization.launch
│   │   │   │   ├── mapping.launch
│   │   │   │   ├── robot_description.launch
│   │   │   │   ├── world.launch
│   │   │   ├── meshes                         # meshes folder for sensors
│   │   │   │   ├── hokuyo.dae
│   │   │   ├── model                          # 3rd party and my_ball model files
│   │   │   │   ├── 3DGEMS
│   │   │   ├── rviz                           # Rviz config file
│   │   │   │   ├── config.rviz
│   │   │   ├── urdf                           # urdf folder for xarco files
│   │   │   │   ├── my_robot.gazebo
│   │   │   │   ├── my_robot.xacro
│   │   │   ├── worlds                         # world folder for world files
│   │   │   │   ├── office.world
│   │   │   ├── CMakeLists.txt                 # compiler instructions
│   │   │   ├── package.xml                    # package info
│   │   ├── teleop_twist_keyboard              # ROS package to send commands to the robot using keyboard or controller to help robot localize
├── images                                     # Video capture of result
│   ├── submit.mp4
│   ├── *.png
├── files                                      # files to create catkin_ws
├── scripts                                    # Scripts to create & run project
│   ├── setupRobot.sh                          # Shell script to setup project files and dependent projects and build
│   ├── launchRobot.sh                         # Shell script to launch simulation - my_robot in Gazebo to load both the world and plugins
│   ├── launchTeleop.sh                        # Shell script to launch keyboard teleop
│   ├── launchMapping.sh                       # Shell script to launch mapping
│   ├── launchLocalization.sh                  # Shell script to launch localization
│   ├── launchMouse.sh                         # Shell script to launch mouse teleop
└──
```

## Build the project  
* Unzip the submission Submit.zip
* Unzip MapMyWorld.zip
* Note: a trimmed version of result is shared on git in images folder
* On terminal
```
  cd MapMyWorld
  chmod +x scripts/*.sh
  ./scripts/setupRobot.sh
```
* This will create catkin_ws folder, copy all the project files from folder 'files' to corresponding catkin_ws folders and build the project
* The resultant catkin_ws folder is shared via catkin_ws.tar.gz

## Run the project  
* On new terminal Launch my_robot in Gazebo to load both the world and plugins  
```
  cd MapMyWorld
  ./scripts/launchRobot.sh
```
* On new terminal Launch keyboard teleop
```
  cd MapMyWorld
  ./scripts/launchTeleop.sh
```
* On new terminal Launch mouse teleop (Optionally)
```
  cd MapMyWorld
  ./scripts/launchMouse.sh
```
* Perform SLAM
```
  cd MapMyWorld
  ./scripts/launchMapping.sh
```
* Use either keyboard or mouse teleop controls to control robot to navigate and perform mapping 

## Results
1. Robot's simulated environment
![alt text][image1]
2. On completing mapping, rviz map
![alt text][image2]
3. Localization launch, rviz map
![alt text][image2]

**RTAB-Map database Analysis**
```
  rtabmap-databaseViewer MapMyWorld/catkin_ws/src/my_robot/database/rtabmap.db
```
4. ![alt text][image4]
5. ![alt text][image5]
6. ![alt text][image6]
7. ![alt text][image7]

**Exporting 3D map as mesh**
8. ![alt text][image8]
9. ![alt text][image9]
10. ![alt text][image10]
11. ![alt text][image11]
12. ![alt text][image12]

All above results are from runs on local VM. I could not view the resultant database on udacity workspace. Got error
```
[ERROR] (2020-12-27 05:50:53.014) DBDriverSqlite3.cpp:398::connectDatabaseQuery() Opened database version (0.20.7) is more recent than rtabmap installed version (0.19.3). Please update rtabmap to new version!
```

**Notes**
1. teleop_tools has keyboard option too. I have not explored it, and have used teleop_twist_keyboard
2. Had to modify with Mem/STMSize value to get correct mapping based off loops
3. Need to improve camera mounting on robot to get better mapping results
4. Reduce the area of model to perform mapping faster. And also improve landmark features