from keras.models import load_model
import numpy as np

from dist.t_es_to_c_es.simple_dn import load_data, weighted_binary_crossentropy, f1_m

MODEL_FILE_PATH = '/home/daniel/heMoji/dist/t_es_to_c_es/model.hdf5'


def convert_vec_to_emojis(vec, labels_to_emojis, only_ones):
    vec = list(vec.reshape(64))
    emojis = " "

    if only_ones:
        for i,v in enumerate(vec):
            if v == 1.0:
                e = labels_to_emojis[i]
                emojis += e

    else:  # model evaluation
        top_5_idxs = np.argsort(vec)[::-1][:5]
        for i in top_5_idxs:
            e = labels_to_emojis[i]
            emojis += e

    return emojis


def get_sample():
    X_train, X_test, Y_train, Y_test, emojis_to_labels, labels_to_emojis = load_data()
    y = None
    # x = X_train[0:1]
    # y = Y_train[0:1]
    # x_emojis = convert_vec_to_emojis(x, labels_to_emojis, only_ones=True)
    # y_emojis = convert_vec_to_emojis(y, labels_to_emojis, only_ones=True)

    with open('evaluate_sample.txt', 'r') as f_evaluate_sample:
        emojis = f_evaluate_sample.readline().split()
        l_labels = []
        for e in emojis:
            l_labels.append(emojis_to_labels[e.decode('utf-8')])
        x = np.zeros(64)
        for i in l_labels:
            x[i] = 1
        x=x.reshape(1, 64)

    x_emojis = convert_vec_to_emojis(x, labels_to_emojis, only_ones=True)
    # y_emojis = convert_vec_to_emojis(y, labels_to_emojis, only_ones=True)

    msg = "x emojis: {0}\n".format(x_emojis.encode('utf-8'))
    print(msg)
    f.writelines(msg)

    # msg = "y emojis: {0}\n".format(y_emojis.encode('utf-8'))
    # print(msg)
    # f.writelines(msg)

    return x, y, emojis_to_labels, labels_to_emojis


if __name__ == '__main__':
    f = open('result.txt', 'w')

    model = load_model(MODEL_FILE_PATH, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy,
                                                        'f1_m': f1_m})
    x, y, emojis_to_labels, labels_to_emojis = get_sample()

    y_hat = model.predict(x)
    result = convert_vec_to_emojis(y_hat, labels_to_emojis, only_ones=False)
    msg = "y_hat emojis: {0}\n".format(result.encode('utf-8'))
    print(msg)
    f.writelines(msg)

    # y_hat_stat = model.evaluate(x,y)

    f.close()
    pass

