#!/bin/bash

DATA_PKL_FILE=$1
VOCAB_FILE=$2
LOGS_DIR=$3
MAXLEN=$4
BATCH_SIZE=$5
EPOCHS=$6

[[ ! -z "$DATA_PKL_FILE" ]] && echo "DATA_PKL_FILE: $DATA_PKL_FILE" || echo "[WARNING] DATA_PKL_FILE arg empty"
[[ ! -z "$VOCAB_FILE" ]] && echo "VOCAB_FILE: $VOCAB_FILE" || echo "[WARNING] VOCAB_FILE arg empty"
[[ ! -z "$LOGS_DIR" ]] && echo "LOGS_DIR: $LOGS_DIR" || echo "[WARNING] LOGS_DIR arg empty"
[[ ! -z "$LOGS_DIR" ]] && echo "LOGS_DIR: $LOGS_DIR" || LOGS_DIR='/home/daniel/heMoji/logs'
[[ ! -z "$MAXLEN" ]] && echo "MAXLEN: $MAXLEN" || echo "[WARNING] MAXLEN arg empty"
[[ ! -z "$BATCH_SIZE" ]] && echo "BATCH_SIZE: $BATCH_SIZE" || echo "[WARNING] BATCH_SIZE arg empty"
[[ ! -z "$EPOCHS" ]] && echo "EPOCHS: $EPOCHS" || echo "[WARNING] EPOCHS arg empty"


# set log subdir
dt=$(date '+%d_%m_%Y_%H_%M_%S');
sd=$LOGS_DIR"/model/$dt/"
mkdir -p $sd
echo "Logging to: $sd"


python src/train_model.py $DATA_PKL_FILE $VOCAB_FILE $sd $MAXLEN $BATCH_SIZE $EPOCHS &> $sd/log.txt