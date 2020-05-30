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


if __name__ == '__main__':
    data_path, out_path = get_args()

    print("Loading model ...")
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    sentok = SentenceTokenizer(vocab, prod=True, wanted_emojis=e2l, uint=32)

    model = load_model(PRETRAINED_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    with open(data_path, 'r') as f:
        print("Predicting text data from {} ...".format(data_path))

        full_out = []
        short_out = []
        lines = f.readlines()
        n_sents = len(lines)

        for i in trange(n_sents):
            line = lines[i]
            line = line.strip()
            tokens = encode_input_sentence(sentok, input_sentence=line)
            if tokens is not None:
                e_scores = model.predict(tokens)[0]
                e_labels = np.argsort(e_scores)
                e_labels_reverse = e_labels[::-1]
                e_labels_reverse_probs = [e_scores[i] for i in e_labels_reverse]
                emojis = [l2e[e].decode('utf-8') for e in e_labels_reverse]
                e_top_labels = e_labels_reverse[:5]  # top
                e_top_labels_scores = e_labels_reverse_probs[:5]  # top
                top_emojis = emojis[:5]

                full_out.append({'input': line,
                                 'emojis': str(emojis),
                                 'probs': str(e_labels_reverse_probs)})
                short_out.append((line, top_emojis))
            else:
                full_out.append({'input': line,
                                 'emojis': 'N/A',
                                 'probs': 'N/A'})
                short_out.append((line, 'N/A'))

    print("Dumping results to {} ...".format(out_path))

    # full out
    with open(out_path + 'out.json', 'w') as f:
        json.dump(full_out, f, indent=0)

    # short out
    with open(out_path + 'out.txt', 'w') as f:
        for i in short_out:
            emojis = "".join([e for e in i[1]])
            f.writelines("{0}: {1}\n".format(i[0], emojis))

    print("Successfully Done !")



