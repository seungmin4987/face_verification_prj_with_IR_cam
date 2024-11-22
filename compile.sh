#!/bin/bash

# OpenCV 3.4.5 경로
OPENCV_DIR=/usr/local/opencv345

# 컴파일 명령
g++ /home/seungmin/opencv_prj/main.cpp -o face_verification \
    -I/home/seungmin/opencv_prj/include \
    -I$OPENCV_DIR/include \
    -L$OPENCV_DIR/lib \
    -Wl,-rpath,$OPENCV_DIR/lib \
    `pkg-config --cflags --libs $OPENCV_DIR/lib/pkgconfig/opencv.pc`

