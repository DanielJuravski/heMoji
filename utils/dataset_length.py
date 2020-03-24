import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE_PATH = '/home/daniel/heMoji/data/data_3G.pkl'


def get_params():
    if len(sys.argv) == 2:
        data_file = sys.argv[1]
    else:
        data_file = DATA_FILE_PATH

    return data_file


def get_stats(data_file):
    d = dict()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        X = data['X']

        samples_len = []
        for s in X:
            samples_len.append(len(s))

        d["l_all"] = samples_len
        d["l_min"] = min(samples_len)
        d["l_max"] = max(samples_len)
        d["l_avg"] = np.average(samples_len)

    return d


def print_stats(stats):
    for k,v in stats.items():
        print("{0}:\t{1}".format(k, str(v)))

    plt.hist(stats["l_all"])#, bins=np.arange(EMOJIS_NUM) - 0.5, facecolor='g')
    plt.show()


if __name__ == '__main__':
    "scan data.pkl xs, get min, max, avg, etc. number of tokens"
    data_file = get_params()
    stats = get_stats(data_file)
    print_stats(stats)