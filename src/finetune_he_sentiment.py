from __future__ import print_function
from __future__ import division
import json
import sys
import numpy as np
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.model_def import hemoji_transfer
from lib.finetuning import load_benchmark, finetune
from src import raw_to_pickle


DATASET_PATH = '/home/daniel/heMoji/datasets/he_sentiment_twitter/token/'
# DATASET_PATH = 'datasets/he_sentiment_twitter/token_data.pkl'
LOGS_DIR = '/home/daniel/heMoji/logs/finetune_he_sentiment_last/'
PRETRAINED_PATH = '/home/daniel/heMoji/data/model.h5'  # this should be a file that created with save_weights cmd
VOCAB_PATH = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'
EPOCHS = 1
EPOCH_SIZE = 100  # relevant when training via batch generator
USE_BATCH_GENERATOR = False
TRANSFER = 'chain-thaw'
EARLY_STOP = False

TIMES = dict()


def get_args(DATASET_PATH, LOGS_DIR, PRETRAINED_PATH, VOCAB_PATH, EPOCHS, TRANSFER):
    parser = argparse.ArgumentParser(description='Sentiment finetuning')
    parser.add_argument('--data', type=str, required=False, default=DATASET_PATH, help='Data (train.tsv, dev.tsv and test.tsv) dir path')
    parser.add_argument('--logs_dir', '--out', type=str, required=False, default=LOGS_DIR, help='Results dir path')
    parser.add_argument('--epochs', type=int, required=False, default=EPOCHS, help='Number of epochs of iterating the data')
    parser.add_argument('--gpu', type=str, required=False, default="-1", help='GPU number to execute on')

    # dev args
    parser_dev = argparse.ArgumentParser(description='Sentiment finetuning', add_help=False)
    parser_dev.add_argument('--model', type=str, required=False, default=PRETRAINED_PATH)
    parser_dev.add_argument('--vocab', type=str, required=False, default=VOCAB_PATH)
    parser_dev.add_argument('--epoch_size', type=int, required=False, default=EPOCH_SIZE)
    parser_dev.add_argument('--train_data_gen', dest='train_data_gen', action='store_true', default=USE_BATCH_GENERATOR)
    parser_dev.add_argument('--early_stop', dest='early_stop', action='store_true', default=EARLY_STOP)
    parser_dev.add_argument('--transfer', type=str, required=False, default=TRANSFER)

    args = parser.parse_known_args()
    args_dev = parser_dev.parse_known_args()

    params = dict()
    params['logs_dir'] = args[0].logs_dir
    params['data_path'] = args[0].data
    params['model_path'] = args_dev[0].model
    params['vocab_path'] = args_dev[0].vocab
    params['epochs'] = args[0].epochs
    params['epoch_size'] = args_dev[0].epoch_size
    params['train_data_gen'] = args_dev[0].train_data_gen
    params['gpu'] = args[0].gpu
    params['early_stop'] = args_dev[0].early_stop
    params['transfer'] = args_dev[0].transfer

    print("params:")
    for k, v in params.items():
        print("{0}:\t{1}".format(k, v))
    print()

    return params


def printTime(key, msg):
    t = datetime.datetime.now()
    print("{0}: {1}".format(msg, t.strftime('%d/%m/%Y_%H:%M:%S')))
    TIMES[key] = t


def save_stats(model, test_acc, logs_dir, train_val_stats, method):
    def makeGraphs(train_acc_list, val_acc_list, train_loss_list, val_loss_list, logs_dir):
        # acc graph
        plt.plot(train_acc_list, label="Train")
        plt.plot(val_acc_list, label="Val")
        plt.gca().legend()
        # plt.show()
        fig_name = logs_dir + "acc.png"
        print("Plotting acc to: {0}".format(fig_name))
        plt.savefig(fig_name)
        plt.close()

        # loss graph
        plt.plot(train_loss_list, label="Train")
        plt.plot(val_loss_list, label="Val")
        plt.gca().legend()
        # plt.show()
        fig_name = logs_dir + "loss.png"
        print("Plotting loss to: {0}".format(fig_name))
        plt.savefig(fig_name)

    if train_val_stats is not None:
        # chain-thaw method
        train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_val_stats
    else:
        if method == 'add-last':
            h = model.model.history.history
        else:
            h = model.history.history
        train_acc_list = h['acc']
        val_acc_list = h['val_acc']
        train_loss_list = h['loss']
        val_loss_list = h['val_loss']

    makeGraphs(train_acc_list, val_acc_list, train_loss_list, val_loss_list, logs_dir)
    with open(logs_dir + 'stats.txt', 'w') as f:
        f.writelines("Test data acc: {0}\n".format(test_acc))
        print("Test data acc: {0}\n".format(test_acc))
        train_time = (TIMES['train_end'] - TIMES['train_start'])  # .strftime('%d/%m/%Y_%H:%M:%S')
        msg = "All train time: " + str(train_time) + '\n'
        f.writelines(msg)


def count_oov(data, vocab):
    train_oov = np.count_nonzero(data['texts'][0] == vocab['CUSTOM_UNKNOWN'])
    test_oov = np.count_nonzero(data['texts'][2] == vocab['CUSTOM_UNKNOWN'])
    train_oov_ratio = train_oov / 28787
    test_oov_ratio = test_oov / 18912

    print("Train tokens OOV ratio: {0} ({1} tokens out of {2})".format(train_oov_ratio, train_oov, 28787))
    print("Test tokens OOV ratio: {0} ({1} tokens out of {2})".format(test_oov_ratio, test_oov, 18912))


def init_hemoji_architecture(nb_classes, data, params, nb_tokens):
    if params['transfer'] == 'add-last':
        model = hemoji_transfer(nb_classes=64, maxlen=data['maxlen'], weight_path=params['model_path'],
                                nb_tokens=nb_tokens, gpu=params['gpu'], exclude_layer_names=[])
    else:
        model = hemoji_transfer(nb_classes, data['maxlen'], params['model_path'], nb_tokens=nb_tokens, gpu=params['gpu'])

    return model


def get_nb_classes(data):
    nb_0 = len(np.unique(data['labels'][0]))
    nb_1 = len(np.unique(data['labels'][1]))
    nb_2 = len(np.unique(data['labels'][2]))

    nb = np.max((nb_0, nb_1, nb_2))

    return nb


def main(params):
    with open(params['vocab_path'], 'r') as f:
        vocab = json.load(f)
        nb_tokens = len(vocab)

    # Load dataset.
    data = load_benchmark(params['data_path']+'/data.pkl', vocab, vocab_uint=32)
    nb_classes = get_nb_classes(data)

    # count OOV
    # count_oov(data, vocab)

    # Set up model and finetune
    model = init_hemoji_architecture(nb_classes, data, params, nb_tokens)
    model.summary()
    printTime(key='train_start', msg="Start Training X,Y data")
    model, test_acc, train_val_stats = finetune(model, data['texts'], data['labels'], nb_classes,
                                                data['batch_size'], method=params['transfer'],
                                                epoch_size=params['epoch_size'], nb_epochs=params['epochs'],
                                                batch_generator=params['train_data_gen'], early_stop=params['early_stop'])
    printTime(key='train_end', msg="End Training X,Y data")

    save_stats(model, test_acc, params['logs_dir'], train_val_stats, method=params['transfer'])

    # save model
    model_path = params["logs_dir"] + "model.hdf5"
    print("Saving model to: {0}".format(model_path))
    model.save(model_path, include_optimizer=True)


if __name__ == '__main__':
    """Finetuning example.

    Trains the heMoji model on the he sentiment tweeter dataset,
    using the 'last', 'chain-thaw' and 'add-last' finetuning method and the accuracy metric.

    The 'last' method (transfer param) does the following:
    0) Load all weights except for the softmax layer. Do not add tokens to the
       vocabulary and do not extend the embedding layer.
    1) Freeze all layers except for the softmax layer.
    2) Train.

    The 'chain-thaw' method (transfer param) does the following:
    0) Load all weights except for the softmax layer. Extend the embedding layer if
       necessary, initialising the new weights with random values.
    1) Freeze every layer except the last (softmax) layer and train it.
    2) Freeze every layer except the first layer and train it.
    3) Freeze every layer except the second etc., until the second last layer.
    4) Unfreeze all layers and train entire model.
    
    The 'add-last' method (my edit) (transfer param) does the following:
    0) Load all weights including the softmax layer.
       Do not add tokens to the vocabulary and do not extend the embedding layer.
    1) Freeze all layers including the softmax layer.
    2) Add MLP layer (32,nb_calsses_ after the softmax layer.
    3) Train. 
    """

    params = get_args(DATASET_PATH, LOGS_DIR, PRETRAINED_PATH, VOCAB_PATH, EPOCHS, TRANSFER)
    raw_to_pickle.process(params['data_path'])
    main(params)
