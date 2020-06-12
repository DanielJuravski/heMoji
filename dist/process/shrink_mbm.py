import pandas as pd

COLS_TO_RM = ["event_plainText_parsed_word",
              "event_plainText_parsed_lemma",
              "event_plainText_parsed_pos",
              "event_plainText_parsed_feats"]
MBM_SRC_FILE_PATH = "/home/daniel/Documents/heMoji_poc/natalie_data/mbm/MBM_text_AllWithSBS.csv"
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm.csv"


def process_mbm(mbm_src):
    mbm = mbm_src

    # drop cols
    for col in COLS_TO_RM:
        print("dropping: {0}".format(col))
        del mbm[col]

    return mbm


if __name__ == '__main__':
    """
    input: mbm file
    output: shrinked mbm file (all cols that appear in COLS_TO_RM will be removed removed)
    """
    mbm_src = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    mbm_target = process_mbm(mbm_src)
    mbm_target.to_csv(MBM_TARGET_FILE_PATH, encoding='utf-8', sep=',')

