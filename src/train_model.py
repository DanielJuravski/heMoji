import sys
import pickle
import json
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

from lib.sentence_tokenizer import SentenceTokenizer
from lib.model_def import hemoji_architecture
from src.emoji2label import e2l


DATA_FILE_PATH = '/home/daniel/heMoji/data/data_mini.pkl'
VOCAB_FILE_PATH = '/home/daniel/heMoji/data/vocabulary.json'

MAXLEN = 80
BATCH_SIZE = 32
EPOCHS = 5


def getArgs():
    params = dict()
    if len(sys.argv) == 7:
        data_file = sys.argv[1]
        vocab_file = sys.argv[2]
        params["logs_dir"] = sys.argv[3]
        params["maxlen"] = int(sys.argv[4])
        params["batch_size"] = int(sys.argv[5])
        params["epochs"] = int(sys.argv[6])
    else:
        print("[WARNING] Using default params")
        data_file = DATA_FILE_PATH
        vocab_file = VOCAB_FILE_PATH
        params["logs_dir"] = "/home/daniel/heMoji/logs/"
        params["maxlen"] = MAXLEN
        params["batch_size"] = BATCH_SIZE
        params["epochs"] = EPOCHS

    print("""\nLoading data file: "{0}"\nLoading vocab file: "{1}"\n""".format(
        data_file, vocab_file))
    for (k,v) in params.iteritems():
        print("param:{0}, value:{1}".format(k,v))
    print("\n")

    return data_file, vocab_file, params


def loadVocab(vocab_file):
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        print("Vocab size is: {0}".format(len(vocab)))

    return vocab


def loadData(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        Y = data['Y']

    return X, Y


def splitData(X, Y):
    st = SentenceTokenizer(vocab, 80, pre_data=True)

    # Split using the default split ratio [0.7, 0.1, 0.2]
    (x_train, x_dev, x_test), (y_train, y_dev, y_test), added = st.split_train_val_test(X, Y)

    # print (x_train, x_dev, x_test)
    # print (y_train, y_dev, y_test)
    # print added
    print(len(x_train), 'train sequences')
    print(len(x_dev), 'test sequences')
    print(len(x_test), 'test sequences')

    return (x_train, x_dev, x_test), (y_train, y_dev, y_test)


def padData(x_train, x_dev, x_test):
    # not sure if necessary, because fixed_length is given in SentenceTokenizer
    x_train = sequence.pad_sequences(x_train)  # , maxlen=maxlen)
    x_dev = sequence.pad_sequences(x_dev)  # , maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test)  # , maxlen=maxlen)
    print('X_train shape:', x_train.shape)
    print('X_dev shape:', x_dev.shape)
    print('X_test shape:', x_test.shape)

    return x_train, x_dev, x_test


def trainModel(vocab, x_train, x_dev, x_test, y_train, y_dev, y_test, params):
    print('Build model...')
    nb_classes = len(e2l)
    vocab_size = len(vocab)

    model = hemoji_architecture(nb_classes=nb_classes, nb_tokens=vocab_size, maxlen=params["maxlen"])
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    h = model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=params["epochs"], validation_data=(x_dev, y_dev))

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=params["batch_size"])
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

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
    Usage: python [DATA_FILE_PATH] [VOCAB_FILE_PATH] [LOGS_DIR] [MAXLEN] [BATCH_SIZE] [EPOCHS] 
    Train heMoji model
    """
    data_file, vocab_file, params = getArgs()
    (X, Y) = loadData(data_file)
    vocab = loadVocab(vocab_file)
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitData(X, Y)
    (x_train, x_dev, x_test) = padData(x_train, x_dev, x_test)

    # model
    h, model, test_loss, test_acc = trainModel(vocab, x_train, x_dev, x_test, y_train, y_dev, y_test, params)

    saveArtifacts(model, h, test_acc, test_loss, params)

