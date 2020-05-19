# ***heMoji*** Docker 🐋 Container

To make it easy for others to use the heMoji model, we release it as a dockerised image which includes an easy-to-use pretrained Keras model and evaluation scripts.


# Quick Start

Get the heMoji docker image (using the below cmd you'll get the latest image).

    docker pull danieljuravski/hemoji
Run a container of that image.

    docker run -it danieljuravski/hemoji
You can now explore some of the functionallity of this container:

## Emoji Predict
Predicting emojis for given a text:

    python emojis_predict.py
Will predict the most suitable emojis for each line of text in the `data/examples.txt` file and dump the results to `data/out.txt` file.

## Sentiment predict
The above sentiment labels were trained over amram et al. (2017) dataset;

> 0: Positive sentiment
> 1: Negative sentiment
> 2: Neutral sentiment

Predicting sentiment for given a text:

    python transfer_predict.py

Will predict the sentiment of each line of text in the `data/amram_2017/examples.txt` file and dump the results to `data/amram_2017/out.txt` file.

## Exit
Run `exit` to exit the container (in the above way of execution, it will automatically stop and exit).



# Advanced Setup
You'll probobly would like to use the model to analyse/fine-tune your own data (saying it is in your  `my_data` dir).

Run a container of above image:

    docker run -it --name hemoji -v $(pwd)/my_data:/my_data danieljuravski/hemoji

where:
- `-it` is for running the container in interactive mode and allocate a tty for it.
- `--name` is for easy access for your container.
-  `-v $(pwd)/my_data:/my_data` is for mounitng you local `my_data` dir into the container.

In any time you can detach out of your container by `CTRL P + Q` and attach it back by `docker exec -it hemoji /bin/bash`.



## Emoji Predict
Predicting emojis for given a text:

    python emojis_predict.py --data /my_data/data.txt --out /my_data/
    
Where:
- `--data` Hebrew sentences file path.
- `--out` Results dir path.

Will predict the most suitable emojis for each line of text in the `my_data/data.txt` file (you may use `data/examples.txt` file as a refference to text file) and dump the results to `my_data/` dir. The results are 2 files `out.txt` and `out.json`. 

> `out.txt`: txt file, where each text is attached with the top 5 suitable emojis that were predicted. 
> `out.json`: json file, where each text is an instance of its' utf-8 decoded string, the 64 predicted emojis by their order and the prediction probability of each emoji.

## Transfer fine-tuning
Beyond the ability to predict the corresponding emoji for a given input text, the model works well as the basis for other sentiment prediction tasks, using transfer learning.
You can fine-tune the model over your data - you should have 3 tsv files (you may use `train.tsv`, `dev.tsv` and `test.tsv` files in `data/amram_2017/` dir as a refference to the desired structure):

    python transfer_finetune.py --data /my_data/ --out /my_data/

Where:
- `--data` Data to finetune on pkl file path.
- `--out` Results dir path.
- `--epochs` Number of epochs of iterating the data.
- `--gpu` GPU number to execute on.

Will create a sentiment model based on your data (and labels). The fine-tuning progress, logs and model will be dumped to `my_data/` dir.  The results and logs are 4 files `model.hdf5`, `stats.txt`, `acc.png` and `loss.png`. 

> `model.hdf5`: fine-tuned model, can be used in the next phase.
> `stats.txt`: some stats including test acc result of the fine-tuning process.
> `acc.png`: train and dev data accuracy plot.
> `loss.png`: train and dev data loss plot.


## Transfer predict
Afterwards you have fine-tuned the model based on your sentiment data, you'll probably want to use it to analyse and predict many others:

    python transfer_predict.py --data /my_data/data.txt --out /my_data/ --model /my_data/model.hdf5

Where:
- `--data` Hebrew sentences file path.
- `--out` Results dir path.
- `--model` Trained finetuned model path.

Will load the model (that was trained in the **Transfer fine-tuning** phase ) and predict the sentiment lables for each line of text in the `my_data/data.txt` file and dump the results to `my_data/` dir. The results are 2 files `out.txt` and `out.json`. 
> `out.txt`: txt file, where each text is attached with predicted sentiment label. 
> `out.json`: json file, where each text is an instance of its' utf-8 decoded string, the predicted sentiment labels by their order and the prediction probability of those labels.


# Credit
The heMoji project was developed by Daniel Juravski at the Bar-Ilan natural language processing lab, as part of a larger project on automatic analysis of text in psychotherapy sessions, in order to gain insights on the psychotherapy process (the project is supervised by Prof. Yoav Goldberg from the computer science department and Dr. Dana Atzil from the Psychology department).