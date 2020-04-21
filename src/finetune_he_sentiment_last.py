from __future__ import print_function
import json
import sys
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
    if '--logs_dir' in sys.argv:
        option_i = sys.argv.index('--logs_dir')
        logs_dir = sys.argv[option_i + 1]
    else:
        logs_dir = LOGS_DIR

    if '--model' in sys.argv:
        option_i = sys.argv.index('--model')
        model_path = sys.argv[option_i + 1]
    else:
        model_path = PRETRAINED_PATH

    if '--vocab' in sys.argv:
        option_i = sys.argv.index('--vocab')
        vocab_path = sys.argv[option_i + 1]
    else:
        vocab_path = VOCAB_PATH

    if '--epochs' in sys.argv:
        option_i = sys.argv.index('--epochs')
        epochs = int(sys.argv[option_i + 1])
    else:
        epochs = EPOCHS

    if '--epoch_size' in sys.argv:
        option_i = sys.argv.index('--epoch_size')
        epoch_size = int(sys.argv[option_i + 1])
    else:
        epoch_size = EPOCH_SIZE

    if '--train_data_gen' in sys.argv:
        train_data_gen = True
    else:
        train_data_gen= USE_BATCH_GENERATOR

    return logs_dir, model_path, vocab_path, epochs, epoch_size, train_data_gen


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


def main(logs_dir, model_path, vocab_path, epochs, epoch_size, train_data_gen):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        nb_tokens = len(vocab)

    # Load dataset.
    data = load_benchmark(DATASET_PATH, vocab, vocab_uint=32)  # TODO: maybe the maxlen should be fixed to 80?

    # Set up model and finetune
    model = hemoji_transfer(nb_classes, data['maxlen'], model_path, nb_tokens=nb_tokens)
    model.summary()
    model, test_acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='last',
                          epoch_size=epoch_size, nb_epochs=epochs, batch_generator=train_data_gen)
    save_stats(model, test_acc, logs_dir)


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
    logs_dir, model_path, vocab_path, epochs, epoch_size, train_data_gen = get_args()
    main(logs_dir, model_path, vocab_path, epochs, epoch_size, train_data_gen)
