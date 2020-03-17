import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, dim, batch_size, dtype, n_channels=1, shuffle=True):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle

        ###
        self.d = data
        self.dtype = dtype
        self.classes = self.d.keys()
        self.n_classes = len(self.classes)
        self.dim = dim
        self.n_samples = 0
        # iterate over all the xs
        for xs in data.values():
            # if shape is 1-d there is only one sample
            if len(xs.shape) == 1:
                self.n_samples += 1
            else:
                self.n_samples += xs.shape[0]
        ###

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, y = self.__data_generation()

        return X, y

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim, self.n_channels))
        X = np.empty((self.batch_size, self.dim), dtype=self.dtype)
        y = np.empty((self.batch_size), dtype=self.dtype)

        labels = np.random.choice(self.classes, self.batch_size, replace=False)

        # Generate data
        for i, label in enumerate(labels):
            # sample ind (sampling in [] range so we need to -1)
            # validate number of xs bigger than 1
            if len(self.d[label].shape) == 1:
                X[i,] = self.d[label]
            else:
                x_ind = np.random.random_integers(len(self.d[label])) - 1
                X[i,] = self.d[label][x_ind]

            # Store class
            y[i] = label

        return X, y


def split_data(X, Y, batch_size):
    "make dict where the key is the label, and the values us vstack of the Xs that correspond to the label"
    assert len(X) == len(Y)

    # make main data dict
    d = dict()
    for x,y in zip(X,Y):
        if y not in d:
            d[y] = x
        else:
            d[y] = np.vstack((d[y], x))

    # make stats
    n_samples = 0
    # iterate over all the xs
    for xs in d.values():
        # if shape is 1-d there is only one sample
        if len(xs.shape) == 1:
            n_samples += 1
        else:
            n_samples += xs.shape[0]
    steps_per_epoch = int(np.floor(n_samples / batch_size))

    return d, steps_per_epoch


def gen(d, batch_size, dtype, dim):
    classes = d.keys()
    # n_classes = len(classes)

    while True:
        X = np.empty((batch_size, dim), dtype=dtype)
        y = np.empty((batch_size), dtype=dtype)

        labels = np.random.choice(classes, batch_size, replace=False)

        # Generate data
        for i, label in enumerate(labels):
            # sample ind (sampling in [] range so we need to -1)
            # validate number of xs bigger than 1
            if len(d[label].shape) == 1:
                X[i,] = d[label]
            else:
                x_ind = np.random.random_integers(len(d[label])) - 1
                X[i,] = d[label][x_ind]

            # Store class
            y[i] = label

        yield X, y