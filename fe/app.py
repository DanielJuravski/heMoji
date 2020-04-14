import streamlit as st
from keras.models import load_model
import json
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
from os.path import expanduser
from time import gmtime, strftime
from PIL import Image
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from lib.sentence_tokenizer import SentenceTokenizer
from lib.attlayer import AttentionWeightedAverage
import SessionState


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
HOME = expanduser("~")
LOGGER_PATH = HOME + '/emoji_predictor.log'


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


def use_example_sentence():
    with open('examples.json') as f:
        conf = json.load(f, encoding='utf-8')
        s1 = conf['s1']
        s2 = conf['s2']
        s3 = conf['s3']
        s4 = conf['s4']
        s5 = conf['s5']
        s6 = conf['s6']
        s7 = conf['s7']

    b1 = st.button(s1)
    b2 = st.button(s2)
    b3 = st.button(s3)
    b4 = st.button(s4)
    b5 = st.button(s5)
    b6 = st.button(s6)
    b7 = st.button(s7)

    if b1:
        return s1
    elif b2:
        return s2
    elif b3:
        return s3
    elif b4:
        return s4
    elif b5:
        return s5
    elif b6:
        return s6
    elif b7:
        return s7
    else:
        return None


def get_input_sentence():
    # user input sentence
    sentence_widget = st.empty()
    sentence_str = sentence_widget.text_input('Insert Hebrew sentence:', key=0)

    # example sentences
    st.write("<p style='font-size:80%;'>Or click on any example sentence below:</p>", unsafe_allow_html=True)
    example_sentence = use_example_sentence()
    # state = SessionState.get(key=0)
    if example_sentence is not None:
        # state.key += 1
        sentence_str = sentence_widget.text_input('Insert Hebrew sentence:', value=example_sentence, key=0)

    return sentence_str


def home_page():
    st.balloons()

    st.title('***heMoji*** Predictor')
    st.subheader('***heMoji*** will try to understand the sentiment of your Hebrew sentence and predict the correspond emoji for it')

    st.sidebar.title("Mode")
    mode = st.sidebar.radio(label="", options=["Basic", "Advanced"])
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    image = Image.open('biu_logo_transparent.png')
    st.sidebar.image(image, width=100)

    sentence = get_input_sentence()

    return mode, sentence


def predict_input_sentence(session, tokens):
    K.set_session(session)
    e_scores = model.predict(tokens)[0]  # there is only 1 macro array since it is the return of the softmax layer
    e_labels = np.argsort(e_scores)  # sort: min --> max
    e_labels_reverse = e_labels[::-1]  # reverse max --> min
    e_labels_reverse_scores = [e_scores[i] for i in e_labels_reverse]  # prob of every label
    emojis = [l2e[e] for e in e_labels_reverse]
    e_top_labels = e_labels_reverse[:TOP_E]  # top
    emojis_top = [l2e[e] for e in e_top_labels]
    e_top_labels_scores = e_labels_reverse_scores[:TOP_E]  # top

    result = pd.DataFrame({'emoji': emojis_top, 'prob': e_top_labels_scores}).T
    log_result = pd.DataFrame({'emoji': emojis, 'emoji_label': e_labels_reverse, 'prob': e_labels_reverse_scores})

    return result, log_result


def style_result(result):
    result = result.style.apply(design_cells, axis=1)
    result = edit_probs(result)

    return result


def encode_input_sentence(input_sentence):
    # encode sentence to tokens
    u_line = [input_sentence]
    try:
        tokens, infos, stats = sentok.tokenize_sentences(u_line)
    except AssertionError as e:
        st.error("I think you've entered invalid sentence, please try another sentence.")
        tokens = None

    return tokens


def logger(input_sentence, tokens, log_result):
    with open(LOGGER_PATH, 'a+') as f:
        t = strftime("%Y_%m_%d-%H:%M:%S", gmtime())
        info = {'time': t,
                'input': input_sentence,
                'input_tokens': tokens.tolist(),
                'prediction': {'emoji': log_result['emoji'].tolist(),
                               'emoji_label': log_result['emoji_label'].tolist(),
                               'emoji_prob': log_result['prob'].tolist()}}
        f.writelines(json.dumps(info))
        f.writelines("\n")


if __name__ == '__main__':
    """
    some pretty UI that loads the model and predicts emoji based on text
    """
    model, session, sentok = loaders()

    mode, input_sentence = home_page()

    if input_sentence:
        tokens = encode_input_sentence(input_sentence)
        if tokens is not None:
            if mode == "Advanced":
                st.write("Input tokens:")
                st.write(tokens)
            result, log_result = predict_input_sentence(session, tokens)
            result_table = style_result(result)

            # display emoji predictions
            st.table(result_table)

            # log session
            logger(input_sentence, tokens, log_result)
            print("Predicted!\n")

