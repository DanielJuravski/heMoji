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
from src.emoji2label import l2edata01 as l2e

DATA_PATH = '/root/heMoji/data/examples.txt'
LOGS_DIR = '/root/heMoji/data/'
PRETRAINED_PATH = '/root/heMoji/model/model.hdf5'
VOCAB_PATH = '/root/heMoji/model/vocab.json'


def get_args():
    parser = argparse.ArgumentParser(description='heMoji Predictor')
    parser.add_argument('--data', type=str, required=False, default=DATA_PATH, help='Hebrew sentences file path')
    parser.add_argument('--out', type=str, required=False, default=LOGS_DIR, help='Results dir path')

    args = parser.parse_args()

    return args.data, args.out


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


def evaluate(model, tokens, top_k=5):
    """
    evaluate the model based on the given tokens
    :param model:
    :param tokens:
    :param top_k: number of emojis to print for short output
    :return:
    """
    e_scores = model.predict(tokens)[0]
    e_labels = np.argsort(e_scores)
    e_labels_reverse = e_labels[::-1]
    e_labels_reverse_probs = [e_scores[i] for i in e_labels_reverse]
    emojis = [l2e[e].decode('utf-8') for e in e_labels_reverse]
    e_top_labels = e_labels_reverse[:top_k]  # top
    e_top_labels_scores = e_labels_reverse_probs[:top_k]  # top
    top_emojis = emojis[:top_k]

    return emojis, top_emojis, e_labels_reverse_probs


def dump_results(line, emojis, emojis_probs, top_emojis):
    full_out = ({'input': line,
                     'emojis': str(emojis),
                     'probs': str(emojis_probs),})

    short_out = (line, top_emojis)

    # full out
    with open(out_path + 'out.json', 'a') as f:
        json.dump(full_out, f, indent=None)
        f.writelines('\n')

    # short out
    with open(out_path + 'out.txt', 'a') as f:
        emojis = "".join([e for e in short_out[1]])
        f.writelines("{0}: {1}\n".format(short_out[0], emojis))


def init_out_files(out_path):
    # full out
    with open(out_path + 'out.json', 'w') as f:
        pass

    # short out
    with open(out_path + 'out.txt', 'w') as f:
        pass


if __name__ == '__main__':
    data_path, out_path = get_args()

    print("Loading model ...")
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    sentok = SentenceTokenizer(vocab, prod=True, wanted_emojis=e2l, uint=32)

    model = load_model(PRETRAINED_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

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
                emojis, top_emojis, emojis_probs = evaluate(model, tokens)
                dump_results(line, emojis, emojis_probs, top_emojis)
            else:
                dump_results(line, 'N/A', 'N/A', 'N/A')

    print("Results were dumped to {}".format(out_path))

    print("Successfully Done !")



