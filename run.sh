#!/usr/bin/env bash

# Add cuda SM for Jetson TX2, you can change to your proper archs.
# It seems speedup compiling
export SMS=62

make clean build -j 4

make run 2>/dev/null

#ARGS="-input=data/color.png"
