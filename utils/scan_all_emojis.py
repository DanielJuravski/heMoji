import emoji
import sys
from collections import defaultdict, OrderedDict
import operator


def get_params():
    if len(sys.argv) < 2:
        tweets_file = '/home/daniel/heMoji/data/hebrew_tweets_3G_mini.tsv'
    else:
        tweets_file = sys.argv[1]

    return tweets_file


def extract_emojis(data):
    emojis_freq = defaultdict(int)
    for line in data:
        u_line = unicode(line, 'utf-8')
        for c in u_line:
            if c in emoji.UNICODE_EMOJI:
                emojis_freq[c] += 1

    sorted_emojis_freq = OrderedDict(sorted(emojis_freq.items(), key=operator.itemgetter(1), reverse=True))

    return sorted_emojis_freq


def load_file(tweets_file):
    with open(tweets_file, 'r') as f:
        data = f.readlines()

    return data


def save(emojis_freq):
    with open('emojis_freq.txt', 'w') as f:
        for k,v in emojis_freq.iteritems():
            f.writelines("{0}: {1}\n".format(k.encode('utf-8'), v))


if __name__ == '__main__':
    """
    Load file where every line is string tweet
    Calculate which emojis are in the data and its freq
    """
    tweets_file = get_params()
    data = load_file(tweets_file)
    emojis_freq = extract_emojis(data)
    save(emojis_freq)

    print(emojis_freq)