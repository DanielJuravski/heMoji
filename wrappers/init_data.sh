#!/bin/bash

TWEETS_TEXT_FILE=$1
VOCAB_FILE=$2
DATA_PKL_FILE=$3

[[ ! -z "$TWEETS_TEXT_FILE" ]] && echo "TWEETS_TEXT_FILE: $TWEETS_TEXT_FILE" || echo "[WARNING] TWEETS_TEXT_FILE arg empty"
[[ ! -z "$VOCAB_FILE" ]] && echo "VOCAB_FILE: $VOCAB_FILE" || echo "[WARNING] VOCAB_FILE arg empty"
[[ ! -z "$DATA_PKL_FILE" ]] && echo "DATA_PKL_FILE: $DATA_PKL_FILE" || echo "[WARNING] DATA_PKL_FILE arg empty"


echo Start making vocab file
python src/make_vocab.py $TWEETS_TEXT_FILE $VOCAB_FILE &> logs/vocab.log
echo Done making vocab file

echo Start making dataset pkl file
python src/make_dataset.py $TWEETS_TEXT_FILE $DATA_PKL_FILE &> logs/dataset.log
echo Done making dataset pkl file
