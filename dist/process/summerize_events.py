import pandas as pd
from copy import deepcopy
import math
from collections import OrderedDict

from add_hemojis import get_emojis_keys


MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis.csv"
DYAD = "tzoa2112"
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis_" + DYAD + "_sum.csv"
SBS_FEATURES_FILE_PATH = "/home/daniel/Documents/heMoji_poc/natalie_data/SBS_Features_18032020.csv"

SPEAKER_MAP = {
    'Client': 'c_',
    'Therapist': 't_'
}


def summerize_moments(mbm):
    # cols to keep/summerize
    # cols_to_summerize are based on cols_to_keep but duplicate for client and therapist summerization
    emojis = get_emojis_keys()
    cols_to_keep = emojis + ['positive_v1', 'negative_v1']

    c_emojis = get_emojis_keys(prefix='c_')
    t_emojis = get_emojis_keys(prefix='t_')
    cols_to_summerize = ['c_positive_v1', 'c_negative_v1', 't_positive_v1', 't_negative_v1'] + c_emojis + t_emojis

    dyad_mbm = mbm.loc[mbm[u't_init']+mbm[u'c_code'] == DYAD]

    # create summerized object
    sbs_data = OrderedDict()
    for index, row in dyad_mbm.iterrows():
        if row['transcription_hard_key'] not in sbs_data:
            s_init_data = {k: 0 for k in cols_to_summerize}
            sbs_data[row['transcription_hard_key']] = deepcopy(s_init_data)

        # summerize only Client and Therapist moments, not Annotator moments
        speaker = row['event_speaker']
        if speaker != 'Client' and speaker != 'Therapist':
            continue

        # iterate over the cols in 'cols_to_keep' and summerize them (based on the speaker)
        for k in cols_to_keep:
            v = row[k]
            if not math.isnan(v):
                sbs_data[row['transcription_hard_key']][SPEAKER_MAP[speaker]+k] += v

    # sbs_data from dict to list to df
    sbs_data_list = []
    for k, v in sbs_data.iteritems():
        v['transcription_hard_key'] = k
        sbs_data_list.append(v)
    sbs_dataframe = pd.DataFrame(sbs_data_list)

    # set 'transcription_hard_key' col to be the first
    cols = ['transcription_hard_key'] + [col for col in sbs_dataframe if col != 'transcription_hard_key']
    sbs_dataframe = sbs_dataframe[cols]

    return sbs_dataframe


def append_features(sbs_data):
    # load feature file
    features = pd.read_csv(SBS_FEATURES_FILE_PATH)
    # which src features to append
    feats_to_append = ['c_a_poms_calmness',
                      'c_a_poms_anger',
                      'c_a_poms_sad',
                      'c_a_poms_contentment',
                      'c_a_poms_anxiety',
                      'c_a_poms_vigor']

    # iterate over target sbs and append for each 'transcription_hard_key' the features above
    feats_list = []
    for index, row in sbs_data.iterrows():
        transcription_hard_key_value = row[u'transcription_hard_key']
        feats = {f: 0 for f in feats_to_append}
        for k in feats_to_append:
            try:
                val = float(features.loc[features[u'transcription_hard_key'] == transcription_hard_key_value][k])
            except ValueError:
                val = 'n/a'
            feats[k] = val
        feats_list.append(feats)

    # convert to dataframe
    feats_dataframe = pd.DataFrame(feats_list)

    # append feats_dataframe to the src one
    sbs_data = pd.concat([sbs_data, feats_dataframe], axis=1)

    return sbs_data


if __name__ == '__main__':
    """
    input: mbm file
    output: mpm (not anymore...) file of specific DYAD where every row is:
        - a session summerized info (hemoji only) based on the mbm info, features are seperated to client and therapist
        - negative & positive counters
        - poms
    """
    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    sbs_data = summerize_moments(mbm)
    sbs_data = append_features(sbs_data)
    sbs_data.to_csv(MBM_TARGET_FILE_PATH, encoding='utf-8', sep=',')

