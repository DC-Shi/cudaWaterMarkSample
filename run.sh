#!/usr/bin/env bash

# Add cuda SM for Jetson TX2, you can change to your proper archs.
# It seems speedup compiling
export SMS=62

make clean build

make run ARGS="-input=data/Lena.pgm"


if [ $? -eq 0 ]; then
    echo OK, converting pgm files into png for easier view.
    python3 ConvertImageToPng.py --folder ./data
else
    echo Run failed.
fi