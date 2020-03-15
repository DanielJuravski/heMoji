import matplotlib
matplotlib.use('Agg')
import sys
import pickle
import json
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import datetime

from lib.sentence_tokenizer import SentenceTokenizer
from lib.model_def import hemoji_architecture
from src.emoji2label import deepe2l as e2l


DATA_FILE_PATH = '/home/daniel/heMoji/data/data_mini.pkl'
VOCAB_FILE_PATH = '/home/daniel/heMoji/data/vocabulary.json'

MAXLEN = 80
BATCH_SIZE = 32
EPOCHS = 3
UINT = 16
DROPOUT_EMB = 0.0
DROPOUT_FINAL = 0.0

TIMES = dict()


def getArgs():
    params = dict()

    if '--data' in sys.argv:
        option_i = sys.argv.index('--data')
        data_file = sys.argv[option_i + 1]
    else:
        data_file = DATA_FILE_PATH

    if '--vocab' in sys.argv:
        option_i = sys.argv.index('--vocab')
        vocab_file = sys.argv[option_i + 1]
    else:
        vocab_file = VOCAB_FILE_PATH

    if '--logs_dir' in sys.argv:
        option_i = sys.argv.index('--logs_dir')
        params["logs_dir"] = sys.argv[option_i + 1]
    else:
        params["logs_dir"] = "/home/daniel/heMoji/logs/"

    if '--maxlen' in sys.argv:
        option_i = sys.argv.index('--maxlen')
        params["maxlen"] = int(sys.argv[option_i + 1])
    else:
        params["maxlen"] = MAXLEN

    if '--batch_size' in sys.argv:
        option_i = sys.argv.index('--batch_size')
        params["batch_size"] = int(sys.argv[option_i + 1])
    else:
        params["batch_size"] = BATCH_SIZE

    if '--epochs' in sys.argv:
        option_i = sys.argv.index('--epochs')
        params["epochs"] = int(sys.argv[option_i + 1])
    else:
        params["epochs"] = EPOCHS

    if '--uint' in sys.argv:
        option_i = int(sys.argv.index('--uint'))
        params["uint"] = sys.argv[option_i + 1]
    else:
        params["uint"] = UINT

    if '--embed_dropout_rate' in sys.argv:
        option_i = int(sys.argv.index('--embed_dropout_rate'))
        params["embed_dropout_rate"] = float(sys.argv[option_i + 1])
    else:
        params["embed_dropout_rate"] = DROPOUT_EMB

    if '--final_dropout_rate' in sys.argv:
        option_i = int(sys.argv.index('--final_dropout_rate'))
        params["final_dropout_rate"] = float(sys.argv[option_i + 1])
    else:
        params["final_dropout_rate"] = DROPOUT_FINAL

    print("""\nLoading data file: "{0}"\nLoading vocab file: "{1}"\n""".format(data_file, vocab_file))

    for (k,v) in params.iteritems():
        print("param:{0}, value:{1}".format(k,v))
    print("\n")

    return data_file, vocab_file, params


def loadVocab(vocab_file):
    printTime(key='load_vocab_start', msg="Start loading vocab")
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        print("Vocab size is: {0}".format(len(vocab)))

    return vocab


def printTime(key, msg):
    t = datetime.datetime.now()
    print("{0}: {1}".format(msg, t.strftime('%d/%m/%Y_%H:%M:%S')))
    TIMES[key] = t


def loadData(data_file):
    printTime(key='load_data_start', msg="Start loading X,Y data")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        Y = data['Y']

    return X, Y


def splitData(X, Y, params):
    printTime(key='split_tokenize_start', msg="Start splitting and tokenizing X,Y data")

    st = SentenceTokenizer(vocab, 80, pre_data=True, uint=params["uint"])

    # Split using the default split ratio [0.7, 0.1, 0.2]
    (x_train, x_dev, x_test), (y_train, y_dev, y_test), added = st.split_train_val_test(X, Y)

    # print (x_train, x_dev, x_test)
    # print (y_train, y_dev, y_test)
    # print added
    print(len(x_train), 'train sequences')
    print(len(x_dev), 'test sequences')
    print(len(x_test), 'test sequences')

    return (x_train, x_dev, x_test), (y_train, y_dev, y_test)


def padData(x_train, x_dev, x_test, params):
    printTime(key='pad_start', msg="Start padding X,Y data")

    # not sure if necessary, because fixed_length is given in SentenceTokenizer
    x_train = sequence.pad_sequences(x_train, maxlen=params["maxlen"])
    x_dev = sequence.pad_sequences(x_dev, maxlen=params["maxlen"])
    x_test = sequence.pad_sequences(x_test, maxlen=params["maxlen"])
    print('X_train shape:', x_train.shape)
    print('X_dev shape:', x_dev.shape)
    print('X_test shape:', x_test.shape)

    return x_train, x_dev, x_test


def trainModel(vocab, x_train, x_dev, x_test, y_train, y_dev, y_test, params):
    printTime(key='train_start', msg="Start Training X,Y data")
    print('Build model...')
    nb_classes = len(e2l)
    vocab_size = len(vocab)

    model = hemoji_architecture(nb_classes=nb_classes, nb_tokens=vocab_size, maxlen=params["maxlen"],
                                embed_dropout_rate=params["embed_dropout_rate"],
                                final_dropout_rate=params["final_dropout_rate"])
    model.summary()

    #import tensorflow.compat.v1 as tf
    # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])#,options = run_opts)

    print('Train...')
    h = model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=params["epochs"], validation_data=(x_dev, y_dev))

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=params["batch_size"])
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    printTime(key='train_end', msg="End Training X,Y data")

    return h, model, test_loss, test_acc


def makeGraphs(train_acc, train_loss, val_acc, val_loss, params):
    # acc graph
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.gca().legend()
    # plt.show()
    fig_name = params["logs_dir"] + "acc.png"
    print("Plotting acc to: {0}".format(fig_name))
    plt.savefig(fig_name)
    plt.close()

    # loss graph
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.gca().legend()
    # plt.show()
    fig_name = params["logs_dir"] + "loss.png"
    print("Plotting loss to: {0}".format(fig_name))
    plt.savefig(fig_name)


def saveStats(train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, params):
    stat_file = params["logs_dir"] + "stat.txt"
    print("Printing statistics to: {0}".format(stat_file))
    with open(stat_file, 'w') as f:
        train_acc_str = "Train acc: {}\n".format(train_acc)
        train_loss_str = "Train loss: {}\n".format(train_loss)
        val_acc_str = "Val acc: {}\n".format(val_acc)
        vak_loss_str = "Val loss: {}\n".format(val_loss)
        test_acc_str = "Test acc: {}\n".format(test_acc)
        test_loss_str = "Test loss: {}\n".format(test_loss)

        f.writelines(train_acc_str)
        f.writelines(train_loss_str)
        f.writelines(val_acc_str)
        f.writelines(vak_loss_str)
        f.writelines(test_acc_str)
        f.writelines(test_loss_str)

        # times
        for k,v in TIMES.iteritems():
            msg = k + ":" + v.strftime('%d/%m/%Y_%H:%M:%S') + '\n'
            f.writelines(msg)
        train_time = (TIMES['train_end'] - TIMES['train_start'])#.strftime('%d/%m/%Y_%H:%M:%S')
        msg = "All train time: " + str(train_time) + '\n'
        f.writelines(msg)


def saveArtifacts(model, h, test_acc, test_loss, params):
    train_acc = h.history['acc']
    train_loss = h.history['loss']
    val_acc = h.history['val_acc']
    val_loss = h.history['val_loss']

    # acc/loss graphs
    makeGraphs(train_acc, train_loss, val_acc, val_loss, params)
    # params
    saveStats(train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, params)

    # save model
    model_path = params["logs_dir"] + "model.hdf5"
    print("Saving model to: {0}".format(model_path))
    model.save(model_path)


if __name__ == '__main__':
    """
    Usage: ./wrappers/train_model.sh --data ../data/data_3G.pkl --vocab ../data/vocab_3G.json --logs_dir ../logs/ --maxlen 80 --batch_size 32 --epochs 15 --uint 16 --embed_dropout_rate 0 --final_dropout_rate 0
    Train heMoji model
    """
    data_file, vocab_file, params = getArgs()
    (X, Y) = loadData(data_file)
    vocab = loadVocab(vocab_file)
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitData(X, Y, params)
    (x_train, x_dev, x_test) = padData(x_train, x_dev, x_test, params)

    # model
    h, model, test_loss, test_acc = trainModel(vocab, x_train, x_dev, x_test, y_train, y_dev, y_test, params)

    saveArtifacts(model, h, test_acc, test_loss, params)

    print("Done successfully.")

