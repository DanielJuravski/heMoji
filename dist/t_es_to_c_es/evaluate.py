from keras.models import load_model

from dist.t_es_to_c_es.simple_dn import load_data, weighted_binary_crossentropy

MODEL_FILE_PATH = '/home/daniel/heMoji/dist/t_es_to_c_es/model.hdf5'


def get_sample():
    X_train, X_test, Y_train, Y_test, emojis_to_labels = load_data()

    return X_train[0:1], Y_train[0:1]


if __name__ == '__main__':
    model = load_model(MODEL_FILE_PATH, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    x, y = get_sample()
    print(x)
    print(y)
    y_hat = model.predict(x)
    y_hat_stat = model.evaluate(x,y)

    pass

