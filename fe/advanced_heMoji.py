import streamlit as st
import json

from heMoji import loaders
from heMoji import EmojiUI
from heMoji import style_result


if __name__ == '__main__':
    # basic model initialization
    model, session, sentok = loaders()
    eui = EmojiUI(model, session, sentok)
    input_sentence_str, example_sentence_str = eui.home_page()
    eui.predict(input_sentence_str, example_sentence_str)

    # advanced features
    # print tokens
    st.write("Input tokens:")
    st.write(eui.tokens)

    # take care transcription json input file
    uploaded_file = st.file_uploader("Choose a transcription json file")
    if uploaded_file is not None:
        trans = json.load(uploaded_file)
        for turn in trans["dialog_turns_list"]:
            if turn["speaker"] == "Client":
                for mini_turn in turn["mini_dialog_turn_list"]:
                    if mini_turn["speaker"] == "Client":
                        st.markdown("------------")
                        input_sentence = mini_turn["plainText"]
                        st.write("<p style='font-size:80%;'>Input sentence:</p>", unsafe_allow_html=True)
                        st.write(input_sentence)
                        tokens = eui.encode_input_sentence(input_sentence)
                        if tokens is not None:
                            result, log_result = eui.evaluate_input_sentence(session=session, tokens=tokens)
                            result_table = style_result(result)
                            # display emoji predictions
                            st.write("<p style='font-size:80%;'>Predicted emojis:</p>", unsafe_allow_html=True)
                            st.table(result_table)
