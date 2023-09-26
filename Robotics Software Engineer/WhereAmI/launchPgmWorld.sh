#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
export GAZEBO_RESOURCE_PATH=${DIR}/catkin_ws/src/my_robot/:${GAZEBO_RESOURCE_PATH}
gzserver src/pgm_map_creator/world/office.world
cd ${DIR}
