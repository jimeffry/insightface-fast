#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/lxy/Downloads/DataSet/faces_ms1m_112x112/
#DATA_DIR=../datasets/
NETWORK=r50
JOB=softmax1e3
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
#CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --prefix "$PREFIX" --per-batch-size 32 > "$LOGFILE" 2>&1 &
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir /data/faces_ms1m_112x112/ --network r50 --loss-type 5 --margin-a 0.9 --margin-m 0.4 --margin-b 0.15 --prefix ../models/model-r50-combMargin/model --per-batch-size 128 > ../models/model-r50-combMargin/log 2>&1 &
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 4 --margin-a 0.9 --margin-m 0.15 --margin-b 0.15 --prefix "$PREFIX" --per-batch-size 32  --pretrained "$PREFIX",2  #> "$LOGFILE" 
