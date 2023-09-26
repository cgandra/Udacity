#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:${DIR}/catkin_ws/src/my_robot/model/
roslaunch my_robot world.launch
cd ${DIR}
