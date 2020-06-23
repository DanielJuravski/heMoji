import pandas as pd
from collections import OrderedDict
import numpy as np
import math
import matplotlib.pyplot as plt
import random

from dist.process.add_hemojis import get_emojis_keys
from dist.process.summerize_events import load_emojis_map

MBM_SRC_FILE_PATH = "/home/daniel/heMoji/dist/data/mbm_hemojis.csv"
DYAD = "ogpb0304"
SESSION_DIV = 20
SESSION_DISPLAY = [0, -1]
MBM_TARGET_FILE_PATH = "/home/daniel/heMoji/dist/plots/" + DYAD + ".png"


def split_data_by_date_key(data):
    """
    split data by the session's date.
    :param data:
    :return: list where each cell is a dataframe of the moments of the corresponded session date
    """
    uniuqe_keys = list(OrderedDict.fromkeys(data['transcription_hard_key'].values))
    data_by_date = []
    for k in uniuqe_keys:
        k_data = mbm.loc[mbm[u'transcription_hard_key'] == k]
        data_by_date.append(k_data)

    return data_by_date


def emojis_to_neg_pos(c_emojis):
    emojis_map = load_emojis_map()
    emojis_labels_aggrigate = {e: 0 for e in set(emojis_map.values())}

    for e, v in c_emojis.iteritems():
        emojis_labels_aggrigate[emojis_map[e]] += v

    emojis_labels_sum = sum(emojis_labels_aggrigate.values())

    pos_norm = emojis_labels_aggrigate['pos_emojis'] / emojis_labels_sum
    neg_norm = emojis_labels_aggrigate['neg_emojis'] / emojis_labels_sum

    return pos_norm, neg_norm


def process(data):
    output_all_sessions = {}
    for session_id in SESSION_DISPLAY:
        session_data = data[session_id]

        step_size = (session_data['dialog_turn_number'].max())/SESSION_DIV
        step_size = int(round(step_size))
        groups = range(0, session_data['dialog_turn_number'].max(), step_size)

        output_this_session = {}
        c_emojis_pos_p = []
        c_emojis_neg_p = []

        for g_i in range(len(groups)-1):

            c_emojis = {e: 0 for e in get_emojis_keys()}
            for i in range(groups[g_i], groups[g_i+1]):
                turns_data = session_data.loc[session_data['dialog_turn_number'] == i]
                for _, row in turns_data.iterrows():
                    speaker = row['event_speaker']
                    transcription_hard_key = row['transcription_hard_key']
                    if speaker == 'Client':
                        for e in c_emojis:
                            v = row[e]
                            if not math.isnan(v):
                                c_emojis[e] += v

            p_pos, p_neg = emojis_to_neg_pos(c_emojis)
            c_emojis_pos_p.append(p_pos)
            c_emojis_neg_p.append(p_neg)

        output_this_session['c_norm_pos'] = c_emojis_pos_p
        output_this_session['c_norm_neg'] = c_emojis_neg_p

        output_all_sessions[transcription_hard_key] = output_this_session

    return output_all_sessions


def plot_norms(output_all_sessions):
    fig = plt.figure()
    ax = fig.gca()

    for session_key, stats in output_all_sessions.iteritems():
        pos_curve = stats['c_norm_pos']
        neg_curve = stats['c_norm_neg']
        ls = random.sample(['-', '--', '-.', ':'], 1)[0]

        plt.plot(neg_curve, label=session_key+'_neg', linestyle=ls, color='red')
        plt.plot(pos_curve, label=session_key+'_pos', linestyle=ls, color='green')

    ax.set_xticks(np.arange(0, SESSION_DIV))
    fig.set_size_inches(20, 6)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    """
    input: mbm_hemojis file
    output: plot of the neg and pos probs over the session (divided dynamically) with option to display several sessions
    """
    # load mbm-hemoji data
    mbm = pd.read_csv(MBM_SRC_FILE_PATH, encoding='utf-8', index_col=0)
    # keep only DYAD's data
    dyads_data = mbm.loc[mbm[u't_init']+mbm[u'c_code'] == DYAD]
    # organize into session level (by date)
    data = split_data_by_date_key(dyads_data)

    output_all_sessions = process(data)

    plot_norms(output_all_sessions)
    pass
