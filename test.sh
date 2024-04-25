#!/bin/bash

STUDENT_ID=6797948 STUDENT_NAME="Antoine EDY" python main.py \
-s veri \
-t veri \
-a mobilenet_v3_small \
--root /content \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/mobilenet_v3_small-veri \
--load-weights logs/mobilenet_v3_small-veri/model.pth.tar-2
