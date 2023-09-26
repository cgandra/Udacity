#!/usr/bin/env bash

DIR=$(pwd )

mkdir -p ${DIR}/catkin_ws/src
cd ${DIR}/catkin_ws/src
catkin_init_workspace
catkin_create_pkg my_robot

mkdir ${DIR}/catkin_ws/src/my_robot/launch
mkdir ${DIR}/catkin_ws/src/my_robot/world
mkdir ${DIR}/catkin_ws/src/my_robot/urdf
mkdir ${DIR}/catkin_ws/src/my_robot/meshes

cp ${DIR}/files/myWorld.world ${DIR}/catkin_ws/src/my_robot/world/
cp ${DIR}/files/world.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/robot_description.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/my_robot.xacro ${DIR}/catkin_ws/src/my_robot/urdf/
cp ${DIR}/files/hokuyo.dae ${DIR}/catkin_ws/src/my_robot/meshes/
cp ${DIR}/files/my_robot.gazebo ${DIR}/catkin_ws/src/my_robot/urdf/
cp -r ${DIR}/files/model ${DIR}/catkin_ws/src/my_robot/

cd ${DIR}/catkin_ws/src
catkin_create_pkg ball_chaser roscpp std_msgs message_generation

mkdir ${DIR}/catkin_ws/src/ball_chaser/srv
mkdir ${DIR}/catkin_ws/src/ball_chaser/launch

cp ${DIR}/files/DriveToTarget.srv ${DIR}/catkin_ws/src/ball_chaser/srv/
cp ${DIR}/files/drive_bot.cpp ${DIR}/catkin_ws/src/ball_chaser/src/
cp ${DIR}/files/process_image.cpp ${DIR}/catkin_ws/src/ball_chaser/src/
cp ${DIR}/files/ball_chaser.launch ${DIR}/catkin_ws/src/ball_chaser/launch/
cp -f ${DIR}/files/CMakeLists_bc.txt ${DIR}/catkin_ws/src/ball_chaser/CMakeLists.txt

cd ${DIR}/catkin_ws/
catkin_make
cd ${DIR}
