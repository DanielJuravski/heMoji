import pandas as pd
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split

DATA_FILE_PATH = '/home/daniel/Documents/heMoji_poc/exposure/SD_unite_v1.csv'
TARGET_TRAIN_FILE_PATH = "/home/daniel/heMoji/dist/exposure/data/train.tsv"
TARGET_DEV_FILE_PATH = "/home/daniel/heMoji/dist/exposure/data/dev.tsv"
TARGET_TEST_FILE_PATH = "/home/daniel/heMoji/dist/exposure/data/test.tsv"


def process(data):
    def print_text(sample_text):
        print(sample_text)
        print(len(sample_text.split()))

    with open('map_labels', 'r') as f:
        lines = f.readlines()
        labels_map = {}
        for line in lines:
            k,v = line.split()
            labels_map[k] = v

    n1_samples = []
    n2_samples = []

    for index, row in data.iterrows():
        try:
            sample_text = row['event_plaintext'].encode('utf-8')
            if len(sample_text.split()) == 1:
                continue
        except AttributeError:
            continue
        ITSD = row['ITSD']
        NITSD = row['NITSD']
        ELSE = row['ELSE']

        if ITSD >= 2:
            print_text(sample_text)
            n1_samples.append(sample_text)

        if NITSD >= 2:
            # print_text(sample_text)
            n1_samples.append(sample_text)
        if ELSE >= 2:
            if 2 < len(sample_text.split()) < 60:
                # print_text(sample_text)
                n2_samples.append(sample_text)

    return n1_samples, n2_samples


def create_dataset_files(pos_samples, neg_samples):
    # shuffle samples
    shuffle(pos_samples)
    shuffle(neg_samples)

    # equalize numbers of samples of both classes
    min_len = np.min((len(pos_samples), len(neg_samples)))
    pos_samples = pos_samples[:min_len]
    neg_samples = neg_samples[:min_len]
    print("number of samples: {0} (pos), {1} (neg)".format(len(pos_samples), len(neg_samples)))

    # make X, y
    X = pos_samples + neg_samples
    y = np.concatenate((np.zeros(len(pos_samples), dtype=int), np.ones(len(neg_samples), dtype=int)))

    # split train:0.7 | dev:0.1 | test:0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.2, random_state=1)

    # print some stats
    print("Number of turns in train: {}".format(len(y_train)))
    print("Number of turns in dev: {}".format(len(y_dev)))
    print("Number of turns in test: {}".format(len(y_test)))
    print

    # dump to files
    with open(TARGET_TRAIN_FILE_PATH, 'w') as f:
        for x, y in zip(X_train, y_train):
            f.writelines("{0}\t{1}\n".format(x, y))
    with open(TARGET_DEV_FILE_PATH, 'w') as f:
        for x, y in zip(X_dev, y_dev):
            f.writelines("{0}\t{1}\n".format(x, y))
    with open(TARGET_TEST_FILE_PATH, 'w') as f:
        for x, y in zip(X_test, y_test):
            f.writelines("{0}\t{1}\n".format(x, y))


if __name__ == '__main__':
    """
    load DATA_FILE_PATH and create train, dev and test tsv files where the labels are:
    - look at 'process' function
    for hemoji finetuning
    """
    data = pd.read_csv(DATA_FILE_PATH, encoding='utf-8', index_col=0)
    n1_samples, n2_samples = process(data)
    create_dataset_files(n1_samples, n2_samples)