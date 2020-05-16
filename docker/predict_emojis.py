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
    print("Loading model ...")
    with open('/heMoji/model/vocab.json', 'r') as f:
    # with open('/home/daniel/heMoji/data/vocab_500G_rare5_data01.json', 'r') as f:
        vocab = json.load(f)

    sentok = SentenceTokenizer(vocab, prod=True, wanted_emojis=e2l, uint=32)

    model = load_model('/heMoji/model/model.hdf5',
    # model = load_model('/home/daniel/Downloads/500G_data01-100K_128_80_rare5_De05_Df05_epochs10_generatorBatch_cce.hdf5',
                       custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})

    with open('/data/predict/sents.txt', 'r') as f:
    # with open('sents.txt', 'r') as f:
        full_out = []
        short_out = []
        lines = f.readlines()
        n_sents = len(lines)
        for i, line in enumerate(lines):
            print("Predicting ... ({0}/{1})".format(i, n_sents))
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

    print("Dumping results ...")

    # full out
    with open('/data/predict/out.json', 'w') as f:
    # with open('out.json', 'w') as f:
        json.dump(full_out, f, indent=0)

    # short out
    with open('/data/predict/out.txt', 'w') as f:
    # with open('out.txt', 'w') as f:
        for i in short_out:
            emojis = "".join([e for e in i[1]])
            f.writelines("{0}: {1}\n".format(i[0], emojis))


