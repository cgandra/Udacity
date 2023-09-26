#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
cd ${DIR}
