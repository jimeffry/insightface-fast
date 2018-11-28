#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/data/Face_Reg_dataset/faces_ms1m_112x112/

NETWORK=m2
JOB=SoftMax
MODELDIR="../models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log2"
LRSTEPS='240000,360000,440000'
CUDA_VISIBLE_DEVICES='2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 128 --lr-steps "$LRSTEPS" --margin-s 32.0 --margin-m 0.1 --ckpt 2 --emb-size 128 --fc7-wd-mult 10.0 --wd 0.00004 --max-steps 140002 --display 1000 --verbose 10000  --lr 0.05 --pretrained "$PREFIX",11 > "$LOGFILE" 
#CUDA_VISIBLE_DEVICES='2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --end-epoch 500000 --margin-a 0.9 --margin-m 0.4 --margin-b 0.15 --prefix "$PREFIX" --#pretrained "$PREFIX",15  --display 1000 --verbose 20000 --per-batch-size 64 --lr 0.00005 --ckpt 2 > "$LOGFILE" 
#CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 5 --data-dir ../datasets/faces_ms1m_112x112  --prefix ../model-r100
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 5 --end-epoch 500000 --margin-a 1.0 --margin-m 0.0 --margin-b 0 --prefix "$PREFIX" --pretrained #"$PREFIX",3 --display 1000 --verbose 10000  --lr 0.0001  --ckpt 2 --margin-s 1.0 --per-batch-size 64 > "$LOGFILE" 

