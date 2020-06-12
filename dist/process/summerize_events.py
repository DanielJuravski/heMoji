import pandas as pd
import requests
import json
from copy import deepcopy
import math
from collections import OrderedDict

from add_hemojis import get_emojis_keys


MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis_iviw0976.csv"
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis_iviw0976_sum.csv"


def summerize_moments(mbm):
    # cols to keep
    emojis = get_emojis_keys()
    cols_to_keep = emojis + ['positive_v1', 'negative_v1']

    sbs_data = OrderedDict()

    for index, row in mbm.iterrows():
        if row['transcription_hard_key'] not in sbs_data:
            s_init_data = {k: 0 for k in cols_to_keep}
            sbs_data[row['transcription_hard_key']] = deepcopy(s_init_data)
        for k in cols_to_keep:
            v = row[k]
            if not math.isnan(v):
                sbs_data[row['transcription_hard_key']][k] += v

    # sbs_data from dict to list to df
    sbs_data_list = []
    for k, v in sbs_data.iteritems():
        v['transcription_hard_key'] = k
        sbs_data_list.append(v)
    sbs_dataframe = pd.DataFrame(sbs_data_list)

    cols = ['transcription_hard_key'] + [col for col in sbs_dataframe if col != 'transcription_hard_key']
    sbs_dataframe = sbs_dataframe[cols]

    return sbs_dataframe


if __name__ == '__main__':
    """
    input: mbm file
    output: mpm (not anymore...) file where every row is a session summerized info (hemoji only) based on the mbm info
    """
    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    sbs_data = summerize_moments(mbm)
    sbs_data.to_csv(MBM_TARGET_FILE_PATH, encoding='utf-8', sep=',')

