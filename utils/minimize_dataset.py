import sys
import pickle
from collections import defaultdict

LARGE_DATA = '/home/daniel/heMoji/data/data.pkl'
SMALL_DATA = '/home/daniel/heMoji/data/data_mini.pkl'
N = 10


def get_args():
    if len(sys.argv) == 4:
        large_data_path = sys.argv[1]
        small_data_path = sys.argv[2]
        n = sys.argv[3]
    else:
        large_data_path = LARGE_DATA
        small_data_path = SMALL_DATA
        n = N

    return large_data_path, small_data_path, n


def small_data_ready(classes_instances):
    """check the number instances of each class
    if every class has n instances, stop the large sampling"""
    if len(classes_instances) == 0:  # if in the first iteration when the dict is empty full will be True,
        # we imidiatly exit the loop since there are not any k,v
        full = False
    else:
        full = True
    for k,v in classes_instances.items():
        if v < n:
            full = False
            break

    return full


def minimize(large_data_path, n):
    print("minimize dataset ...")
    with open(large_data_path, 'rb') as f:
        data = pickle.load(f)
        X_large = data['X']
        Y_large = data['Y']
        assert len(X_large) == len(Y_large)

    # make small data dict
    X_small = []
    Y_small = []
    classes_instances = defaultdict(int)
    for x, y in zip(X_large, Y_large):
        if not small_data_ready(classes_instances):
            if classes_instances[y] < n:
                X_small.append(x)
                Y_small.append(y)
                classes_instances[y] += 1
            else:
                continue

    d = dict()
    d['X'] = X_small
    d['Y'] = Y_small

    return d


def save(data, small_data_path):
    print("save dataset ...")

    with open(small_data_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    """
    input: large data.pkl file
    output: small data.pkl file
    Logic: scan the large data file,
    count number of classes,
    and sample n samples of each class into new small dataset
    """
    large_data_path, small_data_path, n = get_args()
    data = minimize(large_data_path, n)
    save(data, small_data_path)