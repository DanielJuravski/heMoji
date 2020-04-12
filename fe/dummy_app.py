import streamlit as st


if __name__ == '__main__':
    st.title('***heMoji*** Predictor')
    sentence = st.text_input('Insert Hebrew tweet:')
    if sentence:
        st.write("Your input is: {}".format(sentence))


