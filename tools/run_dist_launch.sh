#!/usr/bin/env bash
# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

set -x

GPUS=$1
RUN_COMMAND=${@:2}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python /export/haokangjiang/code/OW-DETR-main/tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}