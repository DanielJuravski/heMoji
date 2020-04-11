import streamlit as st
from keras.models import load_model
import json
import numpy as np
from keras import backend as K

from lib.sentence_tokenizer import SentenceTokenizer
from lib.attlayer import AttentionWeightedAverage

MODEL_PATH = '/home/daniel/heMoji/data/10K_model.hdf5'
VOCAB_FILE = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'

maxlen=80
data = "data01"
e2l_str = data + "e2l"
l2e_str = "l2e" + data
exec("from src.emoji2label import %s as e2l") % e2l_str
exec("from src.emoji2label import %s as l2e") % l2e_str
TOP_E = len(e2l)


@st.cache(allow_output_mutation=True)
def load_my_vocab():
    with open(VOCAB_FILE, 'r') as f:
        vocab = json.load(f)
        print("Vocab size is: {0}".format(len(vocab)))

    return vocab


@st.cache(allow_output_mutation=True)
def load_my_model():
    model = load_model(MODEL_PATH, custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()

    return model, session


@st.cache(allow_output_mutation=False)
def load_sentok(vocab):
    with open(VOCAB_FILE, 'r') as f:
        vocab1 = json.load(f)
    sentok = SentenceTokenizer(vocab1, maxlen, prod=True, wanted_emojis=e2l, uint=32)

    return sentok


if __name__ == '__main__':
    """
    $ streamlit --log_level debug run hello.py
    """
    vocab = load_my_vocab()
    sentok = SentenceTokenizer(vocab, maxlen, prod=True, wanted_emojis=e2l, uint=32)
    # sentok = load_sentok(vocab)
    model, session = load_my_model()

    st.title('My first app')
    st.write('Insert text:')

    sentence = st.text_input('Input your sentence here:')
    if sentence:
        st.write(sentence)
        print (sentence)

        u_line = [sentence]
        tokens, infos, stats = sentok.tokenize_sentences(u_line)
        st.write(tokens)

        K.set_session(session)
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
                st.write(e_unicode)

                pass

        print("Done!\n")








