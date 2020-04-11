import streamlit as st


if __name__ == '__main__':
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        out = sentence + " is a great input!"
        st.write(out)