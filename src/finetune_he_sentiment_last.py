from __future__ import print_function
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.model_def import hemoji_transfer
from lib.finetuning import load_benchmark, finetune

DATASET_PATH = 'datasets/he_sentiment_twitter/data.pickle'
nb_classes = 3

LOGS_DIR = '/home/daniel/heMoji/logs/finetune_he_sentiment_last/'
PRETRAINED_PATH = '/home/daniel/heMoji/data/500G_data01-30K_128_80_rare5_De05_Df05_epochs30_generatorBatch_cce.h5'  # this should be a file that created with save_weights cmd
VOCAB_PATH = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'
EPOCHS = 5
EPOCH_SIZE = 100  # relevant when training via batch generator
USE_BATCH_GENERATOR = False


def get_args():
    params = dict()

    if '--logs_dir' in sys.argv:
        option_i = sys.argv.index('--logs_dir')
        logs_dir = sys.argv[option_i + 1]
    else:
        logs_dir = LOGS_DIR
    params['logs_dir'] = logs_dir

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

    print("params:")
    for k, v in params.items():
        print("{0}:\t{1}".format(k, v))
    print()

    return params


def save_stats(model, test_acc, logs_dir):
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

    train_acc_list = model.history.history['acc']
    val_acc_list = model.history.history['val_acc']
    train_loss_list = model.history.history['loss']
    val_loss_list = model.history.history['val_loss']

    makeGraphs(train_acc_list, val_acc_list, train_loss_list, val_loss_list, logs_dir)
    with open(logs_dir + 'stats.txt', 'w') as f:
        f.writelines("Test data acc: {0}\n".format(test_acc))
        print("Test data acc: {0}\n".format(test_acc))


def main(params):
    with open(params['vocab_path'], 'r') as f:
        vocab = json.load(f)
        nb_tokens = len(vocab)

    # Load dataset.
    data = load_benchmark(DATASET_PATH, vocab, vocab_uint=32)  # TODO: maybe the maxlen should be fixed to 80?

    # Set up model and finetune
    model = hemoji_transfer(nb_classes, data['maxlen'], params['model_path'], nb_tokens=nb_tokens, gpu=params['gpu'])
    model.summary()
    model, test_acc = finetune(model, data['texts'], data['labels'], nb_classes,
                               data['batch_size'], method='last',
                               epoch_size=params['epoch_size'], nb_epochs=params['epochs'],
                               batch_generator=params['train_data_gen'], early_stop=params['early_stop'])
    save_stats(model, test_acc, params['logs_dir'])


if __name__ == '__main__':
    """Finetuning example.

    Trains the heMoji model on the he sentiment tweeter dataset, using the 'last'
    finetuning method and the accuracy metric.

    The 'last' method does the following:
    0) Load all weights except for the softmax layer. Do not add tokens to the
       vocabulary and do not extend the embedding layer.
    1) Freeze all layers except for the softmax layer.
    2) Train.
    """
    params = get_args()
    main(params)
