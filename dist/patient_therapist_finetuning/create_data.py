import pandas as pd
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


SRC_MBM_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm.csv"
TARGET_TRAIN_FILE_PATH = "/home/daniel/heMoji/dist/patient_therapist_finetuning/data/train.tsv"
TARGET_DEV_FILE_PATH = "/home/daniel/heMoji/dist/patient_therapist_finetuning/data/dev.tsv"
TARGET_TEST_FILE_PATH = "/home/daniel/heMoji/dist/patient_therapist_finetuning/data/test.tsv"


def process(mbm):
    client_turns = []
    therapist_turns = []

    # mbm = mbm.head(1000)
    for index, row in mbm.iterrows():
        # take only Client and Therapist turns
        event_speaker = row['event_speaker']
        if event_speaker == 'Annotator':
            continue

        # extract and validate turn text
        text = row['event_plaintext']
        try:
            text = text.encode('utf-8')
            text_len = len(text.split())
        except AttributeError:
            text = None
            text_len = 0
        if (text is not None) and (text_len > 3) and (text_len < 80):
            # append to corresponded data
            if event_speaker == 'Client':
                client_turns.append(text)
            else:
                therapist_turns.append(text)

    # print some stats
    print("Number of Client extracted turns: {}".format(len(client_turns)))
    print("Number of Therapist extracted turns: {}".format(len(therapist_turns)))

    return client_turns, therapist_turns


def create_dataset_files(client_turns, therapist_turns):
    # shuffle samples
    shuffle(client_turns)
    shuffle(therapist_turns)

    # equalize numbers of samples of both classes
    min_len = np.min((len(client_turns), len(therapist_turns)))
    client_turns = client_turns[:min_len]
    therapist_turns = therapist_turns[:min_len]
    print("Total number of samples: {0} (Clients) + {0} (Therapists) = {1} samples".format(min_len, 2*min_len))

    # make X, y
    X = client_turns + therapist_turns
    y = np.concatenate((np.zeros(min_len, dtype=int), np.ones(min_len, dtype=int)))

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
    load SRC_MBM_FILE_PATH and create train, dev and test tsv files where:
        - only client and therapist texts labeled separately
        - only turn that < 80 words counts
    for hemoji finetuning
    """
    mbm = pd.read_csv(SRC_MBM_FILE_PATH, encoding='utf-8', index_col=0)
    client_turns, therapist_turns = process(mbm)
    create_dataset_files(client_turns, therapist_turns)
