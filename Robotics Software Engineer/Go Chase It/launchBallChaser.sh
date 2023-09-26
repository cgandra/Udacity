#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
roslaunch ball_chaser ball_chaser.launch
cd ${DIR}
