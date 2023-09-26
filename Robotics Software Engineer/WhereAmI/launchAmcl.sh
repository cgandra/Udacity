#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
roslaunch my_robot amcl.launch
cd ${DIR}
