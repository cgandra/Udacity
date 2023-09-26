[//]: # (Image References)

[image1]: images/view1.jpg
[image2]: images/view1Trans.jpg
[image3]: images/view2.jpg
[image4]: images/view2Trans.jpg
[image5]: images/view3.jpg
[image6]: images/view3Trans.jpg

# Build My World
Project 1 of Udacity Robotics Software Engineer Nanodegree Program

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

## Overview  
In this project you'll create a simulation world for all your projects in this Robotics Software Engineer Nanodegree Program
1. Build a single floor wall structure using the **Building Editor** tool in Gazebo. Apply at least one feature, one color, and optionally one texture to your structure. Make sure there's enough space between the walls for a robot to navigate.
2. Model any object of your choice using the **Model Editor** tool in Gazebo. Your model links should be connected with joints.
3. Import your structure and two instances of your model inside an empty **Gazebo World**.
4. Import at least one model from the **Gazebo online library** and implement it in your existing Gazebo world.  
5. Write a C++ **World Plugin** to interact with your world. Your code should display “Welcome to ’s World!” message as soon as you launch the Gazebo world file.  

## Setup Instructions
1. First, enable the GPU on your workspace by clicking Enable GPU.
2. Open the visual desktop by clicking on Go to Desktop. The workspace is best supported on **Google Chrome** and might not load on other browsers.
3. Then, update and upgrade the Workspace image to get the latest features of Gazebo. To do so, open a terminal, and write the following statement:

    ``` sudo apt-get update && sudo apt-get upgrade -y ```

    **Note**: Remember to update and upgrade your image after each reboot since these updates(or any package that you install) are not permanent. Ignore any error you get while upgrading
  
4. Build and run your code.  

## Project Description  
Directory Structure  
```
BuildMyWorld                       # Build My World Project 
├── model                          # Model files 
│   ├── myWorld
│   │   ├── model.config
│   │   ├── model.sdf
│   ├── myRobot
│   │   ├── model.config
│   │   ├── model.sdf
│   ├── 3DGEMS
│   │   ├── ReadMe.txt
├── script                         # Gazebo World plugin C++ script      
│   ├── myWorld.cpp
├── world                          # Gazebo main World containing models 
│   ├── myWorld.world
├── CMakeLists.txt                 # Link libraries
├── env.sh                         # Shell script to build the plugin and set paths
├── images                         # Screenshots of myWorld from Gazebo tool
│   ├── view*.jpg
└──
```

- [myWorld](/world/myWorld.world): Gazebo world file that includes the models.  
- [myRobot](/model/myRobot): A robot designed in the Model Editor tool of Gazebo.  
- [3DGEMS](/model/3DGEMS): 3rd party models included in myWorld. Pls see readme for more info.
- [Gazebo Plugin](/script/myWorld.cpp): Gazebo world plugin C++ code.  
- [myWorld image](/images/view1.jpg): A screenshot of the final result.  
- [CMakeLists.txt](CMakeLists.txt): File to link the C++ code to libraries.  

## Run the project  
* Unzip the submission
* On terminal
``` 
  cd BuildMyWorld
  chmod +x env.sh
  source ./env.sh
```
* Launch the world file in Gazebo to load both the world and plugin  
```
gazebo world/myWorld.world
```