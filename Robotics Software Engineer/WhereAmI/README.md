# Where Am I
Project 3 of Udacity Robotics Software Engineer Nanodegree Program

https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/WhereAmI/images/submit_15fps.mp4

<video width="480" height="320" controls="controls">
  <source src="images/submit_15fps.mp4" type="video/mp4">
</video>

## Overview  
In this project you'll utilize ROS AMCL package to accurately localize a mobile robot inside a map in the Gazebo simulation environments
1. Create a ROS package that launches a custom robot model in a custom Gazebo world
2. Utilize the ROS AMCL package and the Tele-Operation / Navigation Stack to localize the robot
3. Explore, add, and tune specific parameters corresponding to each package to achieve the best possible localization results

## Setup Instructions
1. First, enable the GPU on your workspace by clicking Enable GPU.
2. Open the visual desktop by clicking on Go to Desktop. The workspace is best supported on **Google Chrome** and might not load on other browsers.
3. Then, update and upgrade the Workspace image to get the latest features of Gazebo. To do so, open a terminal, and write the following statement:

    ``` sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116    ``` 

    ``` sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654    ``` 

    ``` wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -    ``` 

    ``` sudo apt-get update && sudo apt-get upgrade -y ```

    **Note**: Remember to update and upgrade your image after each reboot since these updates(or any package that you install) are not permanent. Ignore any error you get while upgrading

4. Install Dependencies

    ``` sudo apt-get install --only-upgrade ros-kinetic*    ``` 

    ``` sudo apt-get install ros-kinetic-navigation    ``` 

    ``` sudo apt-get install libignition-math2-dev protobuf-compiler    ``` 

5. Build and run your code.  


## Submission Description  
Directory Structure  
```
WhereAmI                                      # Where Am I Project
├── catkin_ws                                  # Catkin workspace
├── images                                     # Video capture of result
│   ├── submit_15fps.mp4
├── files                                      # files to creae catkin_ws
│   ├── model                                  # 3rd party and my_ball model files
│   │   ├── 3DGEMS
│   ├── amcl.launch
│   ├── config.rviz
│   ├── costmap_common_params.yaml
│   ├── hokuyo.dae
│   ├── map.yaml
│   ├── my_robot.gazebo
│   ├── my_robot.xacro
│   ├── office.world
│   ├── officePgm.world
│   ├── robot_description.launch
│   ├── world.launch
├── setupRobot.sh                              # Shell script to setup project files and dependent projects and build
├── launchPgmWorld.sh                          # Shell script to run gzerver with the map file
├── launchPgmCreate.sh                         # Shell script to launch the request_publisher node
├── launchRobot.sh                             # Shell script to launch simulation - my_robot in Gazebo to load both the world and plugins
├── launchAmcl.sh                              # Shell script to launch amcl
├── launchTeleop.sh                            # Shell script to launch teleop
└──
```

## Build the project  
* Unzip the submission Submit.zip
* Unzip WhereAmI.zip
* On terminal
``` 
  cd WhereAmI
  chmod +x *.sh
  ./setupRobot.sh
```
* This will create catkin_ws folder, copy all the project files from folder 'files' to corresponding catkin_ws folders and build the project
* The resultant catkin_ws folder is shared via catkin_ws.tar.gz

## Project Description  
Directory Structure  
```
.WhereAmI                                      # Where Am I Project
├── catkin_ws                                  # Catkin workspace
│   ├── src
│   │   ├── my_robot                           # my_robot package        
│   │   │   ├── config                         # config folder for robot navigation parameters preset 
│   │   │   │   ├── base_local_planner_params.yaml
│   │   │   │   ├── costmap_common_params.yaml
│   │   │   │   ├── global_costmap_params.yaml
│   │   │   │   ├── local_costmap_params.yaml
│   │   │   ├── launch                         # launch folder for launch files   
│   │   │   │   ├── amcl.launch
│   │   │   │   ├── robot_description.launch
│   │   │   │   ├── world.launch
│   │   │   ├── maps                           # Localization maps folder for map created by pgm_map_creator
│   │   │   │   ├── map.pgm
│   │   │   │   ├── map.yaml
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
│   │   ├── pgm_map_creator                    # ROS package to create localization map of world
│   │   ├── teleop_twist_keyboard              # ROS package to send commands to the robot using keyboard or controller to help robot localize
├── images                                     # Video capture of result
│   ├── submit.mp4
├── files                                      # files to creae catkin_ws
├── setupRobot.sh                              # Shell script to setup project files and dependent projects and build
├── launchPgmWorld.sh                          # Shell script to run gzerver with the map file
├── launchPgmCreate.sh                         # Shell script to launch the request_publisher node
├── launchRobot.sh                             # Shell script to launch simulation - my_robot in Gazebo to load both the world and plugins
├── launchAmcl.sh                              # Shell script to launch amcl
├── launchTeleop.sh                            # Shell script to launch teleop
└──
```

## Run the project  
* Create localization map
  * On terminal
    ``` 
    cd WhereAmI
    ./launchPgmWorld.sh
    ```
  * On another terminal
    ```
    cd WhereAmI
    ./launchPgmCreate.sh
    ```
* On new terminal Launch my_robot in Gazebo to load both the world and plugins  
```
./launchRobot.sh
```
* On new terminal Launch acml
```
./launchAcml.sh
```
* On new terminal Launch teleop
```
./launchTeleop.sh
```
* Use Rviz '2D Nav Goal' and teleop controls to set navigation goals, control robot to localize
