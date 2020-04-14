import streamlit as st
from keras.models import load_model
import json
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from lib.sentence_tokenizer import SentenceTokenizer
from lib.attlayer import AttentionWeightedAverage


with open('config.json') as f:
    conf = json.load(f, encoding='utf-8')
maxlen=80
data = "data01"
e2l_str = data + "e2l"
l2e_str = "l2e" + data
exec("from src.emoji2label import %s as e2l") % e2l_str
exec("from src.emoji2label import %s as l2e") % l2e_str
l2e = l2e  # let's have the 'Unresolved' error once and last here
e2l = e2l  # let's have the 'Unresolved' error once and last here
TOP_E = 5  # len(e2l)


@st.cache(allow_output_mutation=True)
def load_my_vocab():
    with open(conf['vocab'], 'r') as f:
        vocab = json.load(f)
        print("Vocab size is: {0}".format(len(vocab)))

    return vocab


@st.cache(allow_output_mutation=True)
def load_heMoji_model():
    # do not load the model to GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = load_model(conf['model'], custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()

    return model, session


def design_cells(val):
    """
    design the result DF that the first line (eomjis) will be large
    and the second line (probs) will be green color
    :param val:
    :return:
    """
    # ugly way to get if it is emoji or prob-val
    if val[0][0] != ".":
        # emoji
        return ['font-size:20pt'] * TOP_E
    else:
        # prob
        return ['color: green'] * TOP_E


def edit_probs(result):
    """
    1. round to 4 digits after dot
    2. remove leading zero
    0.1234567 --> .1234
    :param result:
    :return:
    """
    for i in range(TOP_E):
        p = result.data[i][1]
        p = round(p, 4)
        p_str = str(p)[1:]
        result.data[i][1] = p_str

    return result


def loaders():
    vocab = load_my_vocab()
    sentok = SentenceTokenizer(vocab, maxlen, prod=True, wanted_emojis=e2l, uint=32)
    model, session = load_heMoji_model()

    return model, session, sentok


def page_home():
    st.title('***heMoji*** Predictor')
    st.subheader('***heMoji*** will try to understand the sentiment of your Hebrew sentence and predict the correspond emoji for it')

    st.sidebar.title("Mode")
    mode = st.sidebar.radio(label="", options=["Basic", "Advanced"])

    sentence = st.text_input('Insert Hebrew phrase:')

    return mode, sentence


def predict_input_sentence():
    K.set_session(session)
    e_scores = model.predict(tokens)[0]  # there is only 1 macro array since it is the return of the softmax layer
    e_labels = np.argsort(e_scores)  # sort: min --> max
    e_labels_reverse = e_labels[::-1]  # reverse max --> min
    e_labels_reverse_scores = [e_scores[i] for i in e_labels_reverse]  # prob of every label
    e_top_labels = e_labels_reverse[:TOP_E]  # top
    e_top = [l2e[e] for e in e_top_labels]
    e_top_labels_scores = e_labels_reverse_scores[:TOP_E]  # top

    result = pd.DataFrame({'emoji': e_top, 'prob': e_top_labels_scores}).T

    return result


def style_result(result):
    result = result.style.apply(design_cells, axis=1)
    result = edit_probs(result)

    return result


if __name__ == '__main__':
    """
    some pretty UI that loads the model and predicts emoji based on text
    """
    model, session, sentok = loaders()

    mode, input_sentence = page_home()

    if input_sentence:
        # encode sentence to tokens
        u_line = [input_sentence]
        tokens, infos, stats = sentok.tokenize_sentences(u_line)
        if mode == "Advanced":
            st.write("Input tokens:")
            st.write(tokens)

        result = predict_input_sentence()
        result_table = style_result(result)

        st.table(result_table)

        print("Predicted!\n")

