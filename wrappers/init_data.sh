#!/bin/bash

TWEETS_TEXT_FILE=$1
VOCAB_FILE=$2
DATA_PKL_FILE=$3
LOGS_DIR=$4
THRESHOLD=$5

[[ ! -z "$TWEETS_TEXT_FILE" ]] && echo "TWEETS_TEXT_FILE: $TWEETS_TEXT_FILE" || echo "[WARNING] TWEETS_TEXT_FILE arg empty"
[[ ! -z "$VOCAB_FILE" ]] && echo "VOCAB_FILE: $VOCAB_FILE" || echo "[WARNING] VOCAB_FILE arg empty"
[[ ! -z "$DATA_PKL_FILE" ]] && echo "DATA_PKL_FILE: $DATA_PKL_FILE" || echo "[WARNING] DATA_PKL_FILE arg empty"
[[ ! -z "$LOGS_DIR" ]] && echo "LOGS_DIR: $LOGS_DIR" || echo "[WARNING] LOGS_DIR arg empty"
[[ ! -z "$LOGS_DIR" ]] && echo "LOGS_DIR: $LOGS_DIR" || LOGS_DIR='/home/daniel/heMoji/logs'
[[ ! -z "$THRESHOLD" ]] && echo "THRESHOLD: $THRESHOLD" || echo "[WARNING] THRESHOLD arg empty"


echo Start making vocab file
python src/make_vocab.py $TWEETS_TEXT_FILE $VOCAB_FILE $THRESHOLD &> $LOGS_DIR/vocab.log
echo Done making vocab file

echo Start making dataset pkl file
python src/make_dataset.py $TWEETS_TEXT_FILE $DATA_PKL_FILE &> $LOGS_DIR/dataset.log
echo Done making dataset pkl file
