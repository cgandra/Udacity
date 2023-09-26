#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
export GAZEBO_RESOURCE_PATH=${DIR}/catkin_ws/src/my_robot/:${GAZEBO_RESOURCE_PATH}
cp ${DIR}/catkin_ws/src/pgm_map_creator/maps/map.pgm ${DIR}/catkin_ws/src/my_robot/maps/
roslaunch my_robot world.launch
cd ${DIR}
