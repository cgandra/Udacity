#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
rosrun mouse_teleop mouse_teleop.py mouse_vel:=cmd_vel
cd ${DIR}
