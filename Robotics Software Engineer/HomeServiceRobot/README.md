[//]: # (Image References)

[image1]: images/test_slam.png
[image2]: images/test_navigation.png

# Home Service Robot
Project 5 of Udacity Robotics Software Engineer Nanodegree Program

https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/HomeServiceRobot/images/home_service_20fps.mp4

<video width="480" height="320" controls="controls">
  <source src="images/home_service_20fps.mp4" type="video/mp4">
</video>

## Overview  
In this project you'll build a Home Service Robot in ROS

1. SLAM Testing: Manually perform SLAM by teleoperating your robot
   1. Deploy a turtlebot inside your environment 
   2. Control it with keyboard commands
   3. Interface it with a SLAM package
   4. Visualize the map in rviz
2. Localization and Navigation Testing: Pick two different goals and test your robot's ability to reach them and orient itself with respect to them.
   1. Deploy a turtlebot inside your environment
   2. Localize the turtlebot
   3. Visualize the map in rviz
3. Pick Objects: Write a node that will communicate with the ROS navigation stack and autonomously send successive goals for your robot to reach
4. Modeling Virtual Objects: Model a virtual object with markers in rviz. The virtual object is the one being picked and delivered by the robot, thus it should first appear in its pickup zone, and then in its drop off zone once the robot reaches it
   1. Publish the marker at the pickup zone
   2. Pause 5 seconds
   3. Hide the marker
   4. Pause 5 seconds
   5. Publish the marker at the drop off zone
5. Home Service Robot:  Simulate a full home service robot capable of navigating to pick up and deliver virtual objects. To do so, the add_markers and pick_objects node should be communicating. Or, more precisely, the add_markers node should subscribe to your odometry to keep track of your robot pose.
   1. Initially show the marker at the pickup zone
   2. Hide the marker once your robot reaches the pickup zone
   4. Show the marker at the drop off zone once your robot reaches it

## Setup Instructions
1. First, enable the GPU on your workspace by clicking Enable GPU.
2. Open the visual desktop by clicking on Go to Desktop. The workspace is best supported on **Google Chrome** and might not load on other browsers.
3. Then, update and upgrade the Workspace image to get the latest features of Gazebo

    ``` sudo apt-get update && sudo apt-get upgrade -y ```

    **Note**: Remember to update and upgrade your image after each reboot since these updates(or any package that you install) are not permanent. Ignore any error you get while upgrading
4. Install dependencies
``` 
   sudo apt-get install xterm
   sudo apt-get install ros-kinetic-turtlebot*
   pip install rospkg
``` 

**Local VM setup**
1. Setup VM per Udacity instructions
2. Then, update and upgrade the Workspace image to get the latest features of Gazebo. To do so, open a terminal, and write the following statement:
```
   sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
   sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
   wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
   sudo apt-get update && sudo apt-get upgrade -y
``` 
3. Install Dependencies
``` 
   sudo apt-get install --only-upgrade ros-kinetic*
   sudo apt-get install ros-kinetic-rtabmap-ros
   rosdep install --from-paths src --ignore-src -r -y
   pip install rospkg
   sudo apt-get install xterm
   sudo apt-get install --only-upgrade ros-kinetic*
   sudo apt-get install ros-kinetic-rtabmap-ros
   sudo apt-get install ros-kinetic-openslam-gmapping
   sudo apt-get install ros-kinetic-joy
   sudo apt-get install ros-kinetic-turtlebot*
   sudo apt-get install ros-kinetic-amcl
   sudo apt-get install ros-kinetic-move-base
``` 
5. Build and run your code.  

## Project Description  
Directory Structure  
```
.HomeServiceRobot                           # Project
├── catkin_ws                               # Catkin workspace
│   ├── src
│   │   ├── add_markers                     # add_markers package        
│   │   │   ├── src                         
│   │   │   │   ├── add_markers.cpp
│   │   ├── add_markers_test                # add_markers package for testing
│   │   │   ├── src                         # config folder for robot navigation parameters preset
│   │   │   │   ├── add_markers.cpp
│   │   ├── map                             # Saved map created by rtabmap in project 4
│   │   │   ├── map.pgm                      
│   │   │   ├── map.yaml                    
│   │   ├── model                           # 3rd party and my_ball model files
│   │   │   ├── 3DGEMS
│   │   ├── pick_objects                    # pick_objects package        
│   │   │   ├── src                         
│   │   │   │   ├── pick_objects.cpp
│   │   ├── rviz                            # RViz config files
│   │   │   ├── navigation.rviz             
│   │   ├── slam_gmapping                   # Gmapping source
│   │   ├── turtlebot                       # Turtlebot source
│   │   ├── turtlebot_interactions          # Turtlebot source
│   │   ├── turtlebot_simulator             # Turtlebot source
│   │   ├── world                           # world folder for world files
│   │   │   ├── office.world
├── images                                  # Images used in readme
│   ├── add_markers.mp4
│   ├── home_service_20fps.mp4
│   ├── test_navigation_20fps.mp4
│   ├── test_navigation.png
│   ├── test_slam.png
├── files                                      # files to create catkin_ws
├── scripts                                    # Scripts to create & run project
│   ├── add_markers.sh                         
│   ├── home_service.sh                         
│   ├── pick_objects.sh                        
│   ├── setupRobot.sh                       
│   ├── test_navigation.sh                  
│   ├── test_slam.sh                         
├── gmapping_maps                              # Maps generated after manual slam mapping with gmapping
│   │   │   ├── map.pgm                      
│   │   │   ├── map.yaml                    
└──
```

## Build the project  
* Unzip the submission Submit.zip
* Unzip HomeServiceRobot.zip
* Note: a trimmed version of result is shared on git in images folder

* On terminal
``` 
  cd HomeServiceRobot
  chmod +x *.sh
  ./scripts/setupRobot.sh
```
* This will create catkin_ws folder, copy all the project files from folder 'files' to corresponding catkin_ws folders and build the project
* The resultant catkin_ws folder is shared via catkin_ws.zip

## Run the project  
1. SLAM Testing
   * On terminal
   ``` 
    cd HomeServiceRobot
    ./scripts/test_slam.sh
   ```
   * Resultant map image is shown below. Quality of this mapping is not so good primarily because using keyboard to navigate is harder than mouse. And i could not find mouse functionality in turtlebot_teleop. This map is not used in subsequent steps
    ![alt text][image1]

2. Generate maps with slam mapped database of Project 4
   ``` 
    rosrun map_server map_saver -f map 
   ```

3. Localization and Navigation Testing
   * On terminal
   ``` 
    cd HomeServiceRobot
    ./scripts/test_navigation.sh
   ```

   * Resultant map image
    ![alt text][image2]

   <video width="480" height="320" controls="controls">
     <source src="images/test_navigation_20fps.mp4" type="video/mp4">
   </video>

   https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/HomeServiceRobot/images/test_navigation_20fps.mp4

4. Pick Objects
   * On terminal
   ``` 
   cd HomeServiceRobot
   ./scripts/pick_objects.sh
   ```
5. Add Markers Testing
   * On terminal
   ``` 
   cd HomeServiceRobot
   ./scripts/add_markers.sh
   ```
   <video width="480" height="320" controls="controls">
     <source src="images/add_markers.mp4" type="video/mp4">
   </video>

   https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/HomeServiceRobot/images/add_markers.mp4
   
6. Home Service Robot
   * On terminal
   ``` 
   cd HomeServiceRobot
   ./scripts/home_service.sh
   ```
   <video width="480" height="320" controls="controls">
      <source src="images/home_service_20fps.mp4" type="video/mp4">
   </video>

   https://github.com/cgandra/Udacity/blob/main/Robotics%20Software%20Engineer/HomeServiceRobot/images/home_service_20fps.mp4
