from __future__ import print_function
from __future__ import division
import json
import sys
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.model_def import hemoji_transfer
from lib.finetuning import load_benchmark, finetune


# DATASET_PATH = 'datasets/he_sentiment_twitter_tmp/data.pickle'
DATASET_PATH = 'datasets/he_sentiment_twitter/token_data.pkl'
LOGS_DIR = '/home/daniel/heMoji/logs/finetune_he_sentiment_last/'
PRETRAINED_PATH = '/home/daniel/heMoji/data/500G_data01-30K_128_80_rare5_De05_Df05_epochs30_generatorBatch_cce.h5'  # this should be a file that created with save_weights cmd
VOCAB_PATH = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'
EPOCHS = 2
EPOCH_SIZE = 100  # relevant when training via batch generator
USE_BATCH_GENERATOR = False
TRANSFER = 'last'

nb_classes = 3
TIMES = dict()


def get_args():
    params = dict()

    if '--logs_dir' in sys.argv:
        option_i = sys.argv.index('--logs_dir')
        logs_dir = sys.argv[option_i + 1]
    else:
        logs_dir = LOGS_DIR
    params['logs_dir'] = logs_dir

    if '--data' in sys.argv:
        option_i = sys.argv.index('--data')
        data_path = sys.argv[option_i + 1]
    else:
        data_path = DATASET_PATH
    params['data_path'] = data_path

    if '--model' in sys.argv:
        option_i = sys.argv.index('--model')
        model_path = sys.argv[option_i + 1]
    else:
        model_path = PRETRAINED_PATH
    params['model_path'] = model_path

    if '--vocab' in sys.argv:
        option_i = sys.argv.index('--vocab')
        vocab_path = sys.argv[option_i + 1]
    else:
        vocab_path = VOCAB_PATH
    params['vocab_path'] = vocab_path

    if '--epochs' in sys.argv:
        option_i = sys.argv.index('--epochs')
        epochs = int(sys.argv[option_i + 1])
    else:
        epochs = EPOCHS
    params['epochs'] = epochs

    if '--epoch_size' in sys.argv:
        option_i = sys.argv.index('--epoch_size')
        epoch_size = int(sys.argv[option_i + 1])
    else:
        epoch_size = EPOCH_SIZE
    params['epoch_size'] = epoch_size

    if '--train_data_gen' in sys.argv:
        train_data_gen = True
    else:
        train_data_gen= USE_BATCH_GENERATOR
    params['train_data_gen'] = train_data_gen

    if '--gpu' in sys.argv:
        option_i = sys.argv.index('--gpu')
        gpu = sys.argv[option_i + 1]
    else:
        gpu = "-1"
    params['gpu'] = gpu

    if '--early_stop' in sys.argv:
        early_stop = True
    else:
        early_stop= False
    params['early_stop'] = early_stop

    if '--transfer' in sys.argv:
        option_i = sys.argv.index('--transfer')
        transfer = sys.argv[option_i + 1]
    else:
        print("[WARNING] using default --transfer value. You should pass it's value [last/chain-thaw]")
        transfer = TRANSFER
    params['transfer'] = transfer

    print("params:")
    for k, v in params.items():
        print("{0}:\t{1}".format(k, v))
    print()

    return params


def printTime(key, msg):
    t = datetime.datetime.now()
    print("{0}: {1}".format(msg, t.strftime('%d/%m/%Y_%H:%M:%S')))
    TIMES[key] = t


def save_stats(model, test_acc, logs_dir, train_val_stats):
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
        train_acc_list = model.history.history['acc']
        val_acc_list = model.history.history['val_acc']
        train_loss_list = model.history.history['loss']
        val_loss_list = model.history.history['val_loss']

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


def main(params):
    with open(params['vocab_path'], 'r') as f:
        vocab = json.load(f)
        nb_tokens = len(vocab)

    # Load dataset.
    data = load_benchmark(params['data_path'], vocab, vocab_uint=32)

    # count OOV
    # count_oov(data, vocab)

    # Set up model and finetune
    model = hemoji_transfer(nb_classes, data['maxlen'], params['model_path'], nb_tokens=nb_tokens, gpu=params['gpu'])
    model.summary()
    printTime(key='train_start', msg="Start Training X,Y data")
    model, test_acc, train_val_stats = finetune(model, data['texts'], data['labels'], nb_classes,
                                                data['batch_size'], method=params['transfer'],
                                                epoch_size=params['epoch_size'], nb_epochs=params['epochs'],
                                                batch_generator=params['train_data_gen'], early_stop=params['early_stop'])
    printTime(key='train_end', msg="End Training X,Y data")

    save_stats(model, test_acc, params['logs_dir'], train_val_stats)

    # save model
    model_path = params["logs_dir"] + "model.hdf5"
    print("Saving model to: {0}".format(model_path))
    model.save(model_path)


if __name__ == '__main__':
    """Finetuning example.

    Trains the heMoji model on the he sentiment tweeter dataset, using the 'last'
    and 'chain-thaw' finetuning method and the accuracy metric.

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
    """

    params = get_args()
    main(params)
