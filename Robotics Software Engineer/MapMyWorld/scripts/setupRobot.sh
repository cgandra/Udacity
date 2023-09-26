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
mkdir ${DIR}/catkin_ws/src/my_robot/config
mkdir ${DIR}/catkin_ws/src/my_robot/rviz
mkdir ${DIR}/catkin_ws/src/my_robot/database

cd ${DIR}/catkin_ws/src/my_robot/config
wget https://s3-us-west-1.amazonaws.com/udacity-robotics/Resource/where_am_i/config.zip
unzip config.zip
rm config.zip
cd ${DIR}

cp ${DIR}/files/office.world ${DIR}/catkin_ws/src/my_robot/world/
cp ${DIR}/files/mapping.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/localization.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/world.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/robot_description.launch ${DIR}/catkin_ws/src/my_robot/launch/
cp ${DIR}/files/hokuyo.dae ${DIR}/catkin_ws/src/my_robot/meshes/
cp ${DIR}/files/my_robot.xacro ${DIR}/catkin_ws/src/my_robot/urdf/
cp ${DIR}/files/my_robot.gazebo ${DIR}/catkin_ws/src/my_robot/urdf/
cp -r ${DIR}/files/model ${DIR}/catkin_ws/src/my_robot/
cp ${DIR}/files/costmap_common_params.yaml ${DIR}/catkin_ws/src/my_robot/config/
cp ${DIR}/files/config.rviz ${DIR}/catkin_ws/src/my_robot/rviz/

cd ${DIR}/catkin_ws/src
git clone https://github.com/ros-teleop/teleop_twist_keyboard
git clone https://github.com/ros-teleop/teleop_tools --branch kinetic-devel

cd ${DIR}/catkin_ws/
catkin_make
cd ${DIR}
