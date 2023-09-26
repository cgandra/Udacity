#!/usr/bin/env bash

DIR=$(pwd )

mkdir build
cd build/
cmake ../
make

# In order to make gazebo find the plugin
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:${DIR}/build
