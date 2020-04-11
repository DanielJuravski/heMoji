from keras.models import load_model
import keras
import matplotlib
matplotlib.use('Agg')
import sys
import pickle
import json
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import datetime
import numpy as np

from lib.sentence_tokenizer import SentenceTokenizer
from lib.model_def import hemoji_architecture
from lib.attlayer import AttentionWeightedAverage


MODEL_PATH = '/home/daniel/heMoji/data/10K_model.hdf5'
PROD_SEN = '/home/daniel/heMoji/data/prod_sentence.txt'
VOCAB_FILE = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'
maxlen=80
data = "data01"
e2l_str = data + "e2l"
l2e_str = "l2e" + data
exec "from src.emoji2label import %s as e2l" % e2l_str
exec "from src.emoji2label import %s as l2e" % l2e_str
TOP_E = len(e2l)

with open(VOCAB_FILE, 'r') as f:
    vocab = json.load(f)
    print("Vocab size is: {0}".format(len(vocab)))


### prod usage
with open(PROD_SEN, 'r') as f:
    line = f.readline()
    # make it unicode
    u_line = line.decode('utf-8', 'ignore')
    # make it to list (since tokenize_sentences waits for list)
    u_line = [u_line]
st = SentenceTokenizer(vocab, maxlen, prod=True, wanted_emojis=e2l, uint=32)
tokens, infos, stats = st.tokenize_sentences(u_line)
### prod usage


model = load_model(MODEL_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
model.summary()

e_scores = model.predict(tokens)[0]  # there is only 1 macro array since it is the return of the softmax layer
e_labels = np.argsort(e_scores)  # sort: min --> max


e_labels_reverse = e_labels[::-1]  # reverse max --> min
e_labels_reverse_scores = [e_scores[i] for i in e_labels_reverse]  # prob of every label
e_top_labels = e_labels_reverse[:TOP_E]  # top
e_top_labels_scores = e_labels_reverse_scores[:TOP_E]  # top


with open('../data/out.txt', 'w') as f:
    for e, score in zip(e_top_labels, e_top_labels_scores):
        e_unicode = l2e[e]
        line = e_unicode.encode('utf-8') + '\t' + '(' + str(score) + ')'
        f.writelines(line)
        f.writelines('\n')

        pass


print("\nDone!")

