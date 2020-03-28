from keras.models import load_model
import numpy as np
import sys

from lib.attlayer import AttentionWeightedAverage


MODEL_PATH = '/home/daniel/heMoji/logs/model_3G_generatorBatch.hdf5'
OUTPUT_PATH = '/home/daniel/heMoji/logs/'
DATA_TYPE = "deep"


def get_args():
    if len(sys.argv) == 4:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
        data_type = sys.argv[3]
    else:
        model_path = MODEL_PATH
        output_path = OUTPUT_PATH
        data_type = DATA_TYPE

    e2l_str = data_type + "e2l"
    l2e_str = "l2e" + data_type
    exec "from src.emoji2label import %s as e2l" % e2l_str
    exec "from src.emoji2label import %s as l2e" % l2e_str

    return model_path, output_path, data_type, e2l, l2e


def get_model(model_path):
    model = load_model(model_path, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    model.summary()

    return model


def get_softmax_weights(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight.shape)
    softmax_layer_number = 14
    w = model.get_weights()[softmax_layer_number]

    return w


def export_weights(w, output_path, e2l, l2e):
    w = w.T

    # make vectors file
    np.savetxt(output_path + 'vectors.tsv', w, fmt='%.18e', delimiter='\t', newline='\n')

    # build metadata file
    with open(output_path + 'metadata.tsv', 'w') as f:
        f.writelines("Index\tEmoji")
        f.writelines('\n')
        for e in l2e:
            e_unicode = l2e[e]
            line = str(e) + '\t' + e_unicode.encode('utf-8')
            f.writelines(line)
            f.writelines('\n')


if __name__ == '__main__':
    """
    load model, extract its last layer (softmax matrix), exoprt this matrix into #emojis (64) vectors
    ouput is:
    1. vector file
    2. metadata file - emojis index
    """
    model_path, output_path, data_type, e2l, l2e = get_args()
    model = get_model(model_path)
    w = get_softmax_weights(model)
    export_weights(w, output_path, e2l, l2e)
