import pandas as pd
import requests
import json
from tqdm import trange


MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm.csv"
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis_with_probs.csv"

EVAL_TEXT_LEN_80 = True  # False - do not evaluate text that larger 80 tokens


def get_emojis_keys(prefix=""):
    app_url = "http://127.0.0.1:5000/"
    result = requests.get(url=app_url + 'init')

    emojis = json.loads(result.text)[u'emojis'].values()

    # add prefix to col name (optional)
    emojis = [prefix+e for e in emojis]

    return emojis


def is_text_len_to_eval(text):
    if EVAL_TEXT_LEN_80 == True:
        return True
    else:
        return len(text.split()) < 80


def add_hemojis(mbm):
    # mbm = mbm.iloc[184850:188391+1]  # iviw0976
    mbm = mbm.iloc[184850:184850+200]

    emojis = get_emojis_keys()
    app_url = "http://127.0.0.1:5000/"

    hemojis_data = []
    texts = list(mbm['event_plaintext'])

    for i in trange(len(texts)):
        try:
            text = texts[i].encode('utf-8')
        except AttributeError:
            text = None

        if (text is not None) and (is_text_len_to_eval(text)):  # text smaller than 80 words --> predict
            print(text)
            result = requests.get(url=app_url + text)
            try:
                result_emojis = json.loads(result.text)[u'emojis']
                if result_emojis != u'N/A':
                    hemoji_top_5 = {e: 0 for e in emojis}
                    hemoji_top_5[result_emojis[u'0']] = 1
                    hemoji_top_5[result_emojis[u'1']] = 1
                    hemoji_top_5[result_emojis[u'2']] = 1
                    hemoji_top_5[result_emojis[u'3']] = 1
                    hemoji_top_5[result_emojis[u'4']] = 1
                else:  # tokens are invalid --> no hemoji prediction was
                    hemoji_top_5 = {e: 'n/a' for e in emojis}
            except ValueError:
                hemoji_top_5 = {e: 'n/a' for e in emojis}
        else:  # text bigger than 80 words --> do not predict
            hemoji_top_5 = {e: 'n/a' for e in emojis}

        hemojis_data.append(hemoji_top_5)

    # add the new (hemoji) dataframe to the original dataframe
    hemojis_dataframe = pd.DataFrame(hemojis_data)

    mbm.index = (hemojis_dataframe.index)

    mbm_hemojis = pd.concat([mbm, hemojis_dataframe], axis=1)

    return mbm_hemojis


if __name__ == '__main__':
    """
    input: mbm file
    output: mpm file with additional 64 cols, where each col is an emoji label
            and additional 64 cols, where each col is an emoji prob.
            Each row (where its' text is smaller than 80 tokens) will be evaluated and the results will be dumped to
            the relevant cols.
    """

    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    mbm_hemojis = add_hemojis(mbm)
    mbm_hemojis.to_csv(MBM_TARGET_FILE_PATH, encoding='utf-8', sep=',')
