import pandas as pd
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


FUTURE_ANNOTATION_FILE_PATH = '/home/daniel/Documents/heMoji_poc/future_annotation/future_annotation_only.csv'
TARGET_TRAIN_FILE_PATH = "/home/daniel/heMoji/dist/future_annotation/data/train.tsv"
TARGET_DEV_FILE_PATH = "/home/daniel/heMoji/dist/future_annotation/data/dev.tsv"
TARGET_TEST_FILE_PATH = "/home/daniel/heMoji/dist/future_annotation/data/test.tsv"


def process(data):
    with open('map_labels', 'r') as f:
        lines = f.readlines()
        labels_map = {}
        for line in lines:
            k,v = line.split()
            labels_map[k] = v

    pos_samples = []
    neg_samples = []
    for index, row in data.iterrows():
        try:
            sample_cand_label = row['label'].encode('utf-8')
        except AttributeError:
            sample_cand_label = 'nan'
        sample_text = row['event_plaintext'].encode('utf-8')
        if sample_cand_label in labels_map:
            sample_label = labels_map[sample_cand_label]
            if sample_label == '0':
                pos_samples.append(sample_text)
            elif sample_label == '1':
                neg_samples.append(sample_text)

    return pos_samples, neg_samples


def create_dataset_files(pos_samples, neg_samples):
    # shuffle samples
    shuffle(pos_samples)
    shuffle(neg_samples)

    # equalize numbers of samples of both classes
    # min_len = np.min((len(client_turns), len(therapist_turns)))
    # client_turns = client_turns[:min_len]
    # therapist_turns = therapist_turns[:min_len]
    print("number of samples: {0} (pos), {1} (neg)".format(len(pos_samples), len(neg_samples)))

    # make X, y
    X = pos_samples + neg_samples
    y = np.concatenate((np.zeros(len(pos_samples), dtype=int), np.ones(len(neg_samples), dtype=int)))

    # split train:0.7 | dev:0.1 | test:0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.33, random_state=1)

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
    load FUTURE_ANNOTATION_FILE_PATH and create train, dev and test tsv files where the labels are:
        - future neg
        - future pos
    for hemoji finetuning
    """
    data = pd.read_csv(FUTURE_ANNOTATION_FILE_PATH, encoding='utf-8', index_col=0)
    pos_samples, neg_samples = process(data)
    create_dataset_files(pos_samples, neg_samples)