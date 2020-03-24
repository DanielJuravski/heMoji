from keras.models import load_model
import numpy as np

from lib.attlayer import AttentionWeightedAverage


MODEL_PATH = '/home/daniel/heMoji/logs/model_3G_generatorBatch.hdf5'
OUTPUT_PATH = '/home/daniel/heMoji/logs/'
DATA_TYPE = "deep"

e2l_str = DATA_TYPE + "e2l"
l2e_str = "l2e" + DATA_TYPE
exec "from src.emoji2label import %s as e2l" % e2l_str
exec "from src.emoji2label import %s as l2e" % l2e_str


def get_model():
    model = load_model(MODEL_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    model.summary()

    return model


def get_softmax_weights(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight.shape)
    w = model.get_weights()[14]

    return w


def export_weights(w):
    w = w.T

    # make vectors file
    np.savetxt(OUTPUT_PATH + 'vectors.tsv', w, fmt='%.18e', delimiter='\t', newline='\n')

    # build metadata file
    with open(OUTPUT_PATH + 'metadata.tsv', 'w') as f:
        f.writelines("Index\tEmoji")
        f.writelines('\n')
        for e in l2e:
            e_unicode = l2e[e]
            line = str(e) + '\t' + e_unicode.encode('utf-8')
            f.writelines(line)
            f.writelines('\n')


if __name__ == '__main__':
    model = get_model()
    w = get_softmax_weights(model)
    export_weights(w)
