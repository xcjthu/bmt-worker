#!/bin/bash
PORT=$1
GPU=$2
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT} \
    train.py -c config/DR/dureader.config \
    -g ${GPU} \
    2>&1 | tee log/dureader/dureader.log
