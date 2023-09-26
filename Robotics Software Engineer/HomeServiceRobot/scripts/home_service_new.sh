#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
export GAZEBO_RESOURCE_PATH=$(pwd)/src/:${GAZEBO_RESOURCE_PATH}
export TURTLEBOT_GAZEBO_WORLD_FILE=$(pwd)/src/world/office.world
export TURTLEBOT_GAZEBO_MAP_FILE=$(pwd)/src/map/map.yaml

xterm  -e  " roslaunch turtlebot_gazebo turtlebot_world.launch" & 
sleep 5
xterm  -e  " roslaunch turtlebot_gazebo amcl_demo.launch" & 
sleep 5
xterm  -e  " roslaunch turtlebot_rviz_launchers view_navigation.launch" &
sleep 5
xterm  -e  " rosrun pick_objects_new pick_objects_new"  &
sleep 5
xterm  -e  " rosrun add_markers_new add_markers_new" &

cd ${DIR}
