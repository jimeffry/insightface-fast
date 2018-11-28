#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_ENABLE_GPU_P2P=0

DATA_DIR=/data/Face_Reg_dataset/ms_prison_n/

NETWORK=r100
JOB=SINEV2
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log10"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u new_train_insight.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --end-epoch 500000 --margin-a 0.9 --margin-m 0.5 --margin-b 0.0 --prefix "$PREFIX" --pretrained "$PREFIX",6 --display 1000 --verbose 9000 --per-batch-size 60 --ckpt 2 --lr 0.0002 --easy-margin 0  --lr-steps 5000000,6000000,7000000 > "$LOGFILE" 
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 5 --data-dir ../datasets/faces_ms1m_112x112  --prefix ../model-r100
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 4 --end-epoch 500000 --margin-m 0.2 --prefix "$PREFIX" --pretrained "$PREFIX",22  --display #1000 --verbose 10000  --lr 0.001  --ckpt 2  --margin-s 64.0  --per-batch-size 64 > "$LOGFILE" 
#

