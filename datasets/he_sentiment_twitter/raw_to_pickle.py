import pickle

TYPE = 'morph'  # token/morph
INPUT_TRAIN_FILE = TYPE + '_train.tsv'
INPUT_TEST_FILE = TYPE + '_test.tsv'
OUTPUT_FILE_NAME = TYPE + '_data.pkl'


def load_tsv_data():
    train_X = []
    test_X = []
    train_Y = []
    test_Y = []

    # load train data
    with open('token_train.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            train_X.append(x)
            train_Y.append(int(y))
    # load test data
    with open('token_test.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            test_X.append(x)
            test_Y.append(int(y))

    return train_X, train_Y, test_X, test_Y


def generate_data_obj(train_X, train_Y, test_X, test_Y):
    data = {}

    # texts
    train_X = [x.decode('utf-8') for x in train_X]
    test_X = [x.decode('utf-8') for x in test_X]
    texts = train_X + test_X
    data['texts'] = texts

    # info
    labels = train_Y + test_Y
    labels = [{'label': y} for y in labels]
    data['info'] = labels

    # train_ind
    train_size = len(train_Y)
    train_ind = range(train_size)
    data['train_ind'] = train_ind

    # test_ind
    test_size = len(test_Y)
    test_ind = range(test_size)
    data['test_ind'] = test_ind

    # val_ind
    # add dummy val (take the first 100 from test)
    # this is done because the heMoji tuning expects for train,val,test and here we got train,test only
    data['val_ind'] = test_ind[:100]

    return data


def dump_data(data):
    with open(OUTPUT_FILE_NAME, 'w') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_tsv_data()
    data = generate_data_obj(train_X, train_Y, test_X, test_Y)
    dump_data(data)