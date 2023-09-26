#!/usr/bin/env bash

DIR=$(pwd )

mkdir -p ${DIR}/catkin_ws/src
cd ${DIR}/catkin_ws/src
catkin_init_workspace
catkin_create_pkg my_robot

mkdir ${DIR}/catkin_ws/src/map
mkdir ${DIR}/catkin_ws/src/world
mkdir ${DIR}/catkin_ws/src/rviz

cp ${DIR}/files/office.world ${DIR}/catkin_ws/src/world/
cp -r ${DIR}/files/model ${DIR}/catkin_ws/src/
cp ${DIR}/files/map.pgm ${DIR}/catkin_ws/src/map/
cp ${DIR}/files/map.yaml ${DIR}/catkin_ws/src/map/

catkin_create_pkg pick_objects
mkdir ${DIR}/catkin_ws/src/pick_objects/src
cp ${DIR}/files/pick_objects.cpp ${DIR}/catkin_ws/src/pick_objects/src
cp ${DIR}/files/CMakeLists.txt ${DIR}/catkin_ws/src/pick_objects/

catkin_create_pkg pick_objects_new
mkdir ${DIR}/catkin_ws/src/pick_objects_new/src
cp ${DIR}/files/pick_objects_new.cpp ${DIR}/catkin_ws/src/pick_objects_new/src/pick_objects.cpp
cp ${DIR}/files/CMakeLists_pick_objects_new.txt ${DIR}/catkin_ws/src/pick_objects_new/CMakeLists.txt

catkin_create_pkg add_markers_test
mkdir ${DIR}/catkin_ws/src/add_markers_test/src
cp ${DIR}/files/add_markers_test.cpp ${DIR}/catkin_ws/src/add_markers_test/src/add_markers.cpp
cp ${DIR}/files/CMakeLists_add_marker_test.txt ${DIR}/catkin_ws/src/add_markers_test/CMakeLists.txt

catkin_create_pkg add_markers
mkdir ${DIR}/catkin_ws/src/add_markers/src
cp ${DIR}/files/add_markers.cpp ${DIR}/catkin_ws/src/add_markers/src
cp ${DIR}/files/CMakeLists_add_marker.txt ${DIR}/catkin_ws/src/add_markers/CMakeLists.txt

catkin_create_pkg add_markers_new
mkdir ${DIR}/catkin_ws/src/add_markers_new/src
cp ${DIR}/files/add_markers_new.cpp ${DIR}/catkin_ws/src/add_markers_new/src/add_markers.cpp
cp ${DIR}/files/CMakeLists_add_marker_new.txt ${DIR}/catkin_ws/src/add_markers_new/CMakeLists.txt

cd ${DIR}/catkin_ws/src
git clone https://github.com/ros-perception/slam_gmapping.git --branch hydro-devel
git clone https://github.com/turtlebot/turtlebot.git --branch kinetic
git clone https://github.com/turtlebot/turtlebot_interactions.git --branch indigo
git clone https://github.com/turtlebot/turtlebot_simulator.git --branch indigo

cp ${DIR}/files/navigation.rviz ${DIR}/catkin_ws/src/turtlebot_interactions/turtlebot_rviz_launchers/rviz/

cd ${DIR}/catkin_ws/
catkin_make
cd ${DIR}
