import pickle


def load_tsv_data(DIR_PATH):
    INPUT_TRAIN_FILE = DIR_PATH + '/train.tsv'
    INPUT_DEV_FILE = DIR_PATH + '/dev.tsv'
    INPUT_TEST_FILE = DIR_PATH + '/test.tsv'

    train_X = []
    dev_X = []
    test_X = []
    train_Y = []
    dev_Y = []
    test_Y = []

    # load train data
    with open(INPUT_TRAIN_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            train_X.append(x)
            train_Y.append(int(y))
    # load dev data
    with open(INPUT_DEV_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            dev_X.append(x)
            dev_Y.append(int(y))
    # load test data
    with open(INPUT_TEST_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            test_X.append(x)
            test_Y.append(int(y))

    return train_X, train_Y, dev_X, dev_Y, test_X, test_Y


def generate_data_obj(train_X, train_Y, dev_X, dev_Y, test_X, test_Y):
    data = {}

    # texts
    train_X = [x.decode('utf-8') for x in train_X]
    dev_X = [x.decode('utf-8') for x in dev_X]
    test_X = [x.decode('utf-8') for x in test_X]
    texts = train_X + dev_X + test_X
    data['texts'] = texts

    # info
    labels = train_Y + dev_Y + test_Y
    labels = [{'label': y} for y in labels]
    data['info'] = labels

    # train_ind
    train_size = len(train_Y)
    train_ind = range(train_size)
    data['train_ind'] = train_ind

    # dev_ind
    dev_size = len(dev_Y)
    dev_ind = range(train_size, train_size + dev_size)
    data['val_ind'] = dev_ind

    # test_ind
    test_size = len(test_Y)
    test_ind = range(train_size + dev_size, train_size + dev_size + test_size)
    data['test_ind'] = test_ind

    return data


def dump_data(data, DATA_PATH):
    OUTPUT_FILE_NAME = DATA_PATH + '/data.pkl'
    with open(OUTPUT_FILE_NAME, 'w') as f:
        pickle.dump(data, f)


def process(DATA_PATH):
    """
    load 3 files (train.tsv, dev.tsv, test.tsv), parse them to pkl object for later fine-tune usage and dump it.
    :param DATA_PATH:
    :return:
    """
    # print("Generating data into pkl object")
    train_X, train_Y, dev_X, dev_Y, test_X, test_Y = load_tsv_data(DATA_PATH)
    data = generate_data_obj(train_X, train_Y, dev_X, dev_Y, test_X, test_Y)
    dump_data(data, DATA_PATH)


if __name__ == '__main__':
    DATA_PATH = '/home/daniel/heMoji/datasets/he_sentiment_twitter/token'
    process(DATA_PATH)

