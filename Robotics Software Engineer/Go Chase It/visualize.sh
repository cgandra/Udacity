#!/usr/bin/env bash

DIR=$(pwd )

cd ${DIR}/catkin_ws/
source devel/setup.bash
rosrun rqt_image_view rqt_image_view
cd ${DIR}
