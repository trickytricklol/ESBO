#! /bin/bash

python main.py \
    --batch_size 32 \
    --model DGCNN \
    --epoch 300 \
    --num_points 1024 \
    --workers 8
