import json
from keras.models import load_model
import numpy as np
from tqdm import trange
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from lib.attlayer import AttentionWeightedAverage
from lib.sentence_tokenizer import SentenceTokenizer
from src.emoji2label import data01e2l as e2l

DATA_PATH = '/root/heMoji/data/amram_2017/examples.txt'
LOGS_DIR = '/root/heMoji/data/amram_2017/'
MODEL_PATH = '/root/heMoji/data/amram_2017/model.hdf5'
VOCAB_PATH = '/root/heMoji/model/vocab.json'


def get_args():
    parser = argparse.ArgumentParser(description='Sentiment Predictor')
    parser.add_argument('--data', type=str, required=False, default=DATA_PATH, help='Hebrew sentences file path')
    parser.add_argument('--out', type=str, required=False, default=LOGS_DIR, help='Results dir path')
    parser.add_argument('--model', type=str, required=False, default=MODEL_PATH, help='Trained finetuned model path')

    args = parser.parse_args()

    return args.data, args.out, args.model


def encode_input_sentence(sentok, input_sentence):
    # encode sentence to tokens
    u_line = [input_sentence.decode('utf-8')]
    try:
        tokens, infos, stats = sentok.tokenize_sentences(u_line)
    except AssertionError as e:
        print e
        print("I think you've entered invalid sentence, please try another sentence.")
        tokens = None

    return tokens


def init_out_files(out_path):
    # full out
    with open(out_path + 'out.json', 'w') as f:
        pass

    # short out
    with open(out_path + 'out.txt', 'w') as f:
        pass


def evaluate(model, tokens):
    c_probs = model.predict(tokens)[0]
    labels = np.argsort(c_probs)
    labels = labels[::-1]
    e_labels_reverse_probs = [c_probs[i] for i in labels]
    c_highest = labels[0]

    return labels, c_highest, e_labels_reverse_probs


def dump_results(line, labels, labels_probs, c_highest):
    full_out = ({'input': line,
                 'emojis': str(labels),
                 'probs': str(labels_probs), })

    short_out = c_highest

    # full out
    with open(out_path + 'out.json', 'a') as f:
        json.dump(full_out, f, indent=None)
        f.writelines('\n')

    # short out
    with open(out_path + 'out.txt', 'a') as f:
        f.writelines("{0}\n".format(short_out))


if __name__ == '__main__':
    data_path, out_path, model_path = get_args()

    print("Loading model ...")
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    model = load_model(model_path, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    sentok = SentenceTokenizer(vocab, prod=True, wanted_emojis=e2l, uint=32, fixed_length=model.input_shape[1])

    init_out_files(out_path)

    with open(data_path, 'r') as f:
        print("Predicting text data from {} ...".format(data_path))

        lines = f.readlines()
        n_sents = len(lines)

        for i in trange(n_sents):
            line = lines[i]
            line = line.strip()
            tokens = encode_input_sentence(sentok, input_sentence=line)
            if tokens is not None:
                labels, c_highest, labels_probs = evaluate(model, tokens)
                dump_results(line, labels, labels_probs, c_highest)
            else:
                dump_results(line, 'N/A', 'N/A', 'N/A')

    print("Results were dumped to {}".format(out_path))

    print("Successfully Done !")



