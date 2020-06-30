import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from dist.process.add_hemojis import get_emojis_keys

DATA_FILE_PATH = '/home/daniel/heMoji/dist/t_es_to_c_es/data.txt'


def load_data():
    def get_emojis_labels(l):
        l_labels = []
        for e in l.split():
            l_labels.append(emojis_to_labels[e.decode('utf-8')])
        return l_labels

    def labels_to_vector(l):
        v = np.zeros(64)
        for i in l:
            v[i] = 1
        return v

    emojis_keys = get_emojis_keys()
    emojis_to_labels = {e: i for i, e in enumerate(emojis_keys)}

    Xs = []
    Ys = []
    with open(DATA_FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y = line.strip('\n').split('\t')
            x = get_emojis_labels(x)
            x = labels_to_vector(x)
            y = get_emojis_labels(y)
            y = labels_to_vector(y)
            Xs.append(x)
            Ys.append(y)

    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)

    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=1)

    return X_train, X_test, Y_train, Y_test, emojis_to_labels




import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

def init_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(4096, input_dim=64, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))

    from keras import backend as K
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))




    # compile the keras model
    model.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy', f1_m])
    
    return model


def train_model(model, X_train, X_test, Y_train, Y_test):
    model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test),
              epochs=40, batch_size=10)

    X_train, X_test, Y_train, Y_test, emojis_to_labels = load_data()

    # x = X_train[0:1]
    # y = Y_train[0:1]
    # y_hat = model.predict(x)
    # y_hat_stat = model.evaluate(x,y)


    model.save('model.hdf5')



if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, emojis_to_labels = load_data()
    model = init_model()
    train_model(model, X_train, X_test, Y_train, Y_test,)
