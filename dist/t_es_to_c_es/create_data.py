import pandas as pd

from dist.process.add_hemojis import get_emojis_keys

MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis.csv"
OUTPUT_DATA_FILE_PATH = '/home/daniel/heMoji/dist/t_es_to_c_es/data.txt'


def iterate_tunrs(mbm):
    """
    for dumping x and y, you should validate:
    1. the client replies immediately after the therapist talks
    2. both the therapist and the client talks in the same session
    3. both the therapist and the client turn's length is 3<l<80
    :param mbm:
    :return:
    """
    Xs = []
    Ys = []
    cand_x, cand_y = [], []
    c_trans_key, t_trans_key = "", ""
    emojis_keys = get_emojis_keys()

    for index, row in mbm.iterrows():
        if row['event_speaker'] == 'Client':
            c_trans_key = row['transcription_hard_key']
            turn_len = row['num_of_words']
            if 3 < turn_len < 80:
                cand_x = row[emojis_keys][row==1].index.tolist()
            else:
                cand_x = []
        elif row['event_speaker'] == 'Therapist':
            t_trans_key = row['transcription_hard_key']
            turn_len = row['num_of_words']
            if cand_x != [] and \
                c_trans_key == t_trans_key and \
                3 < turn_len < 80:
                cand_y = row[emojis_keys][row==1].index.tolist()
                Xs.append(cand_x)
                Ys.append(cand_y)
                cand_x, cand_y = [], []
                c_trans_key, t_trans_key = "", ""
        # annotator
        # else:
        #     cand_x, cand_y = [], []
        #     c_trans_key, t_trans_key = "", ""

    return Xs, Ys


def dump_data(Xs, Ys):
    print("Number of samples: {}".format(len(Xs)))
    assert len(Xs) == len(Ys)

    with open(OUTPUT_DATA_FILE_PATH, 'w') as f:
        for x, y in zip(Xs, Ys):
            x = " ".join([e.encode('utf-8') for e in x])
            y = " ".join([e.encode('utf-8') for e in y])

            line = "{0}\t{1}\n".format(x, y)
            f.writelines(line)


if __name__ == '__main__':
    """
    input: mbm_hemoji file
    output: create data where generally the xs are the therapist emojis (with no order) and the ys are the client emojis (with no order)
    each line in the output file is:
    c_e1 c_e2 c_e3 c_e4 c_e5\tt_e1 t_e2 t_e3 t_e4 t_e5
    """
    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    Xs, Ys = iterate_tunrs(mbm)
    dump_data(Xs, Ys)
