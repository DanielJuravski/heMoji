import json
from keras.models import load_model
import numpy as np
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
    if '--data' in sys.argv:
        option_i = sys.argv.index('--data')
        data_path = sys.argv[option_i + 1]
    else:
        data_path = DATA_PATH

    if '--out' in sys.argv:
        option_i = sys.argv.index('--out')
        out_path = sys.argv[option_i + 1]
    else:
        out_path = LOGS_DIR

    print("Loading model ...")
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    sentok = SentenceTokenizer(vocab, prod=True, wanted_emojis=e2l, uint=32)

    model = load_model(PRETRAINED_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    with open(DATA_PATH, 'r') as f:
        print("Loading text data from {} ...".format(DATA_PATH))

        full_out = []
        short_out = []
        lines = f.readlines()
        n_sents = len(lines)
        for i, line in enumerate(lines):
            sys.stdout.flush()
            sys.stdout.write("Predicting ... {0}/{1} ({2}%)\r".format(i, n_sents, round((i * 1.0 / n_sents) * 100, 2)))
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
    print("Predicting ... {0}/{1} ({2}%)\r".format(n_sents, n_sents, round((n_sents * 1.0 / n_sents) * 100, 2)))

    print("Dumping results to {} ...".format(LOGS_DIR))

    # full out
    with open(LOGS_DIR + 'out.json', 'w') as f:
        json.dump(full_out, f, indent=0)

    # short out
    with open(LOGS_DIR + 'out.txt', 'w') as f:
        for i in short_out:
            emojis = "".join([e for e in i[1]])
            f.writelines("{0}: {1}\n".format(i[0], emojis))

    print("Successfully Done !")



