#!/bin/bash

echo "this is a sentence $1"
th test.lua -snt "$1"
python stackGAN_code/test.py && \
cd display && \
rm -rf gen_pic && \
cd ../cycleGAN_code && \
python generate.py \
--dataroot ../display \
--model_path ../models \
--which_epoch 60 \
--resize_or_crop crop \
