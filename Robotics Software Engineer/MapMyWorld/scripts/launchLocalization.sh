#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
roslaunch my_robot localization.launch
cd ${DIR}
