#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
export GAZEBO_RESOURCE_PATH=${DIR}/catkin_ws/src/my_robot/:${GAZEBO_RESOURCE_PATH}
roslaunch pgm_map_creator request_publisher.launch
cd ${DIR}
