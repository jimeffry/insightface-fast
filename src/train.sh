#!/usr/bin/bash
#export MXNET_CPU_WORKER_NTHREADS=24
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
#export MXNET_ENABLE_GPU_P2P=0

DATA_DIR="/wdc/LXY.data/faces_webface/"

NETWORK=r100
JOB=SINE
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log7"
CUDA_VISIBLE_DEVICES='2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 4 --end-epoch 500000 --margin-a 1.0 \
            --margin-m 0.5 --margin-b 0.0 --prefix "$PREFIX"  --pretrained "$PREFIX",1 --display 1000 --verbose 4000 --per-batch-size 50  --lr 0.0002 \
            --easy-margin 0  --lr-steps 5000000,6000000,7000000  --ckpt 1 > "$LOGFILE"
