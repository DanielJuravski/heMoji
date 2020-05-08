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
import string
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





def loaders():
    vocab = load_my_vocab()
    sentok = SentenceTokenizer(vocab, maxlen, prod=True, wanted_emojis=e2l, uint=32)
    model, session = load_heMoji_model()

    return model, session, sentok


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


def str_contains_en_chars(str):
    contains = False
    for i in str:
        if i in string.lowercase:
            contains = True
            break
        if i in string.uppercase:
            contains = True
            break

    return contains


def style_result(result):
    result = result.style.apply(design_cells, axis=1)
    result = edit_probs(result)

    return result


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


def use_example_sentence():
    # set selectbox location to the right
    st.markdown("""
                    <style>
                    .stSelectbox>selectbox {
                        unicode-bidi:bidi-override;
                        direction: RTL;
                        display: block;
                        margin-left: auto;}
                    </style>
                        """, unsafe_allow_html=True)

    with open('examples.json') as f:
        conf = json.load(f, encoding='utf-8')
        s1 = conf['s1']
        s2 = conf['s2']
        s3 = conf['s3']
        s4 = conf['s4']
        s5 = conf['s5']
        s6 = conf['s6']
        s7 = conf['s7']

    selectbox_label = "Or choose any example sentence below:"

    selected = st.selectbox(selectbox_label, ["", s1, s2, s3, s4, s5, s6, s7])

    return selected


def encode_input_sentence(sentok, input_sentence):
    # encode sentence to tokens
    u_line = [input_sentence]
    try:
        tokens, infos, stats = sentok.tokenize_sentences(u_line)
    except AssertionError as e:
        st.error("I think you've entered invalid sentence, please try another sentence.")
        tokens = None

    return tokens


def evaluate_input_sentence(model, session, tokens):
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


def show_results(model, session, sentok, sentence):
    log_sentence = False

    # # user input sentence
    # if input_sentence_str:
    #     sentence_str = input_sentence_str
    #     log_sentence = True
    #
    # # state = SessionState.get(key=0)
    # if example_sentence_str is not None:
    #     # state.key += 1
    #     sentence_str = example_sentence_str
    #     log_sentence = False

    if sentence:
        tokens = encode_input_sentence(sentok, sentence)
        if tokens is not None:
            result, log_result = evaluate_input_sentence(model, session, tokens)
            result_table = style_result(result)

            # display emoji predictions
            st.markdown("<p style='font-size:80%;'>Predicted emojis:</p>", unsafe_allow_html=True)
            st.table(result_table)

            if log_sentence:
                logger(sentence, tokens, log_result)


def page_home(model, session, sentok):
    st.balloons()
    st.title('***heMoji*** Predictor')
    st.subheader('***heMoji*** will try to detect the sentiment, emotion and sarcasm of your Hebrew sentence and predict the correspond emoji for it')

    # user input sentence
    w_input_sentence_text = st.empty()
    w_input_sentence_warning = st.empty()

    # example sentences
    example_sentence_str = use_example_sentence()

    # set sentence_widget location to the right
    st.markdown("""<style>input {direction: RTL;}</style>""", unsafe_allow_html=True)
    sentence = w_input_sentence_text.text_input('Insert Hebrew sentence:', value=example_sentence_str)

    if str_contains_en_chars(sentence):
        w_input_sentence_warning.warning("Your sentence contains non Hebrew characters, which the predictor doesn't support")

    show_results(model, session, sentok, sentence)



def page_about():
    st.title('About ***heMoji***')
    st.write(
        "The ***heMoji*** model is based on the wonderful deepMoji (Felbo et al., 2017). It is about time to get some emoji predictions over Hebrew text!")
    st.write(
        "The model was trained over millions of tweets that were composed out of Hebrew sentences and included at least one of following 64 predictable emojis:")
    st.write(u'\U0001f601', u'\U0001f602', u'\U0001f605', u'\U0001f604', u'\U0001f607', u'\U0001f609', u'\U0001f608',
             u'\U0001f60b', u'\U0001f60a', u'\U0001f60d', u'\U0001f60c', u'\U0001f60f', u'\U0001f60e', u'\U0001f611',
             u'\U0001f610', u'\U0001f613', u'\U0001f612', u'\U0001f495', u'\U0001f614', u'\U0001f497', u'\U0001f616',
             u'\U0001f499', u'\U0001f618', u'\U0001f61d', u'\U0001f61c', u'\U0001f61e', u'\U0001f621', u'\U0001f623',
             u'\U0001f622', u'\U0001f625', u'\U0001f624', u'\U0001f3a7', u'\U0001f630', u'\U0001f629', u'\U0001f62b',
             u'\U0001f62a', u'\U0001f62d', u'\U0001f49c', u'\U0001f631', u'\U0001f389', u'\U0001f633', u'\u270c',
             u'\U0001f634', u'\U0001f637', u'\U0001f3b6', u'\u263a', u'\u270b', u'\U0001f44a', u'\U0001f648',
             u'\U0001f44b', u'\U0001f64a', u'\U0001f44d', u'\U0001f44c', u'\U0001f64f', u'\U0001f64c', u'\U0001f48b',
             u'\U0001f44f', u'\U0001f525', u'\U0001f44e', u'\u2665', u'\u2764', u'\U0001f4aa', u'\U0001f615',
             u'\U0001f494')


def ui(model, session, sentok):
    # sidebar
    st.sidebar.title('***heMoji***')
    side_bar_str = st.sidebar.radio('', ('Home', 'About'))
    if side_bar_str == 'Home':
        page_home(model, session, sentok)
    elif side_bar_str == 'About':
        page_about()

    # a nice BIU logo
    image = Image.open('biu_logo_transparent.png')
    st.write("")
    st.image(image, width=100)


if __name__ == '__main__':
    """
    some pretty UI that loads the model and predicts emoji based on text
    """

    # TODO: make the selectbow text R2L
    # TODO: clean the selectboe value id input_text was inserted
    # TODO: set up logger (do not log example sentences)

    model, session, sentok = loaders()

    ui(model, session, sentok)

