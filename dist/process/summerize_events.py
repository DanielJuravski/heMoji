import pandas as pd
from copy import deepcopy
import math
from collections import OrderedDict, defaultdict
import numpy as np
import io
import scipy.stats
import datetime

from add_hemojis import get_emojis_keys


MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis.csv"
DYAD = "rrmo9272"
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis_" + DYAD + "_sum.csv"
SBS_FEATURES_FILE_PATH = "/home/daniel/Documents/heMoji_poc/natalie_data/SBS_Features_18032020.csv"
EMOJIS_MAP_FILE_PATH = "emojis.map"

SPEAKER_MAP = {
    'Client': 'c_',
    'Therapist': 't_'
}


def add_normalize_seesion_counters_of(sbs_data, k, v, features):
    sum = np.sum([v[f] for f in features])
    for f in features:
        sbs_data[k]['norm_' + f] = (v[f] / float(sum))

    return sbs_data


def load_emojis_map():
    f = io.open(EMOJIS_MAP_FILE_PATH, mode="r", encoding="utf-8")
    lines = f.readlines()
    emojis_map = dict()
    for line in lines:
        if line[0] == '#':
            continue
        e, cat = line.strip('\n').split('\t')
        emojis_map[e] = cat

    return emojis_map


def add_pos_neg_emojis_counters(sbs_data, features, prefix):
    """
    add columnns of clusternig the emojis counts based on EMOJIS_MAP_FILE_PATH (for example negative vs positive emojis)
    :param sbs_data:
    :param features:
    :param prefix:
    :return:
    """
    emojis_map = load_emojis_map()
    for k, v in sbs_data.iteritems():
        for e in features:
            map_key = prefix + emojis_map[e.strip(prefix)]
            if map_key not in sbs_data[k]:
                sbs_data[k][map_key] = 0
            sbs_data[k][map_key] += v[e]

    return sbs_data


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

    # sum emojis counts to pos/neg groups
    sbs_data = add_pos_neg_emojis_counters(sbs_data, c_emojis, 'c_')
    sbs_data = add_pos_neg_emojis_counters(sbs_data, t_emojis, 't_')

    # add normalized values for client and therapist counters
    for k,v in sbs_data.iteritems():
        # normalize c_emojis
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, c_emojis)

        # normalize t_emojis
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, t_emojis)

        # normalize 'c_positive_v1', 'c_negative_v1'
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['c_positive_v1', 'c_negative_v1'])

        # normalize 't_positive_v1', 't_negative_v1'
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['t_positive_v1', 't_negative_v1'])

        # sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['c_neg_emojis', 'c_pos_emojis'])
        # sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['t_neg_emojis', 't_pos_emojis'])
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['c_neg_emojis', 'c_pos_emojis', 'c_anomal_emojis'])
        sbs_data = add_normalize_seesion_counters_of(sbs_data, k, v, ['t_neg_emojis', 't_pos_emojis', 't_anomal_emojis'])

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
                       'c_a_poms_vigor',
                       'c_a_poms19',  # happy
                       'c_b_ors']

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


def set_norm_emojis_entropy(sbs_data):
    """
    iterate the sessions, for each session - calc the entropy value,
    the entropy value is based over the norm_emojis (64) probabilities.
    entropy both for client and therapist is calculated
    :param sbs_data:
    :return:
    """
    ent_data = []
    ent_keys = ["c_ent", "t_ent"]
    c_emojis = get_emojis_keys(prefix='norm_c_')
    t_emojis = get_emojis_keys(prefix='norm_t_')
    top_dists = 5
    for index, row in sbs_data.iterrows():
        ent = {k: 0 for k in ent_keys}
        # calc ent for client emojis (the values in the df are strings. you need float them first)
        c_ent = scipy.stats.entropy([float(v) for v in row[c_emojis].values][:top_dists])
        ent['c_ent'] = c_ent

        # calc ent for therapist emojis (the values in the df are strings. you need float them first)
        t_ent = scipy.stats.entropy([float(v) for v in row[t_emojis].values][:top_dists])
        ent['t_ent'] = t_ent

        ent_data.append(ent)

    # convert to dataframe
    ent_dataframe = pd.DataFrame(ent_data)

    # append feats_dataframe to the src one
    sbs_data = pd.concat([sbs_data, ent_dataframe], axis=1)

    return sbs_data


def add_date(sbs_data):
    dates_list = []
    for index, row in sbs_data.iterrows():
        transcription_hard_key_value = row[u'transcription_hard_key']
        date_val = transcription_hard_key_value.split('_')[1]
        date = {'date': date_val}
        dates_list.append(date)

    dates_dataframe = pd.DataFrame(dates_list)
    sbs_data = pd.concat([sbs_data, dates_dataframe], axis=1)

    return sbs_data


if __name__ == '__main__':
    """
    input: mbm file
    output: mpm (not anymore...) file of specific DYAD where every row is:
        - a session summerized info (hemoji only) based on the mbm info, features are seperated to client and therapist
        - negative & positive counters
        - poms
        - append clear date value of the session
    """
    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    sbs_data = summerize_moments(mbm)
    # sbs_data = set_norm_emojis_entropy(sbs_data)
    sbs_data = append_features(sbs_data)
    sbs_data = add_date(sbs_data)
    sbs_data.to_csv(MBM_TARGET_FILE_PATH, encoding='utf-8', sep=',')

    print("[OK] dumped to {}".format(MBM_TARGET_FILE_PATH))