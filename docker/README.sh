#!/bin/bash

LC='\033[1;36m'
LP='\033[1;35m'
NC='\033[0m' # No Color

echo
echo -e "${LC}🎉 WELCOME TO HEMOJI CONTAINER 🎉${NC}"
echo
echo -e "${LP}For predicting emojis based on text run:${NC}"
echo -e "${LP}    $ python hemoji_predict.py [options]${NC}"
echo -e "${LP}    Options:${NC}"
echo -e "${LP}      --data\t\tHebrew sentences file path${NC}"
echo -e "${LP}      --out\t\tResults dir path${NC}"
echo
echo -e "${LP}For finetuning heMoji model over another data run:${NC}"
echo -e "${LP}    $ python sentiment_finetune.py [options]${NC}"
echo -e "${LP}    Options:${NC}"
echo -e "${LP}      --data\t\tData to finetune on pkl file path${NC}"
echo -e "${LP}      --out\t\tResults dir path${NC}"
echo -e "${LP}      --epochs\t\tNumber of epochs of iterating the data${NC}"
echo -e "${LP}      --gpu\t\tGPU number to execute on${NC}"
echo
echo -e "${LP}For evaluating finetuned heMoji model over another data run:${NC}"
echo -e "${LP}    $ python sentiment_predict.py [options]${NC}"
echo -e "${LP}    Options:${NC}"
echo -e "${LP}      --data\t\tHebrew sentences file path${NC}"
echo -e "${LP}      --out\t\tResults dir path${NC}"
echo -e "${LP}      --model\t\tTrained finetuned model path${NC}"
echo



