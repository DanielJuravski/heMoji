import codecs
import sys

from lib.create_vocab import VocabBuilder
from lib.word_generator import TweetWordGenerator


TWEETS_TEXT_FILE = '/home/daniel/heMoji/data/hebrew_tweets_3G_mini_test.tsv'
DATA_PKL_FILE = '/home/daniel/heMoji/data/data_mini.pkl'
DATA = "deep"


def getArgs():
    if len(sys.argv) == 4:
        tweets_text_file = sys.argv[1]
        data_file = sys.argv[2]
        data = sys.argv[3]
    else:
        tweets_text_file = TWEETS_TEXT_FILE
        data_file = DATA_PKL_FILE
        data = DATA

    e2l_str = data + "e2l"
    l2e_str = "l2e" + data
    exec "from src.emoji2label import %s as e2l" % e2l_str
    exec "from src.emoji2label import %s as l2e" % l2e_str

    print("""\nLoading tweets text file: "{0}" and Parsing it to data file: "{1}"\n""".format(
        tweets_text_file, data_file))

    return tweets_text_file, data_file, e2l, l2e


def make(tweets_text_file, data_file, e2l, l2e):
    with codecs.open(tweets_text_file, 'rU', 'utf-8') as stream:
        wg = TweetWordGenerator(stream, allow_unicode_text=True, wanted_emojis=e2l)
        vb = VocabBuilder(wg, wanted_emojis=e2l)
        vb.set_tweet_tag()
        vb.save_labels(data_file)


if __name__ == '__main__':
    """
    Creates a dataset of (X,Y) from a tsv file.
    Where X is a non parsed sentence (original sentence from the tweet text file)
    Where Y is a label (class number) of some emoji
    """
    tweets_text_file, data_file, e2l, l2e = getArgs()
    make(tweets_text_file, data_file, e2l, l2e)
