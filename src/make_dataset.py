import codecs
import sys

from lib.create_vocab import VocabBuilder
from lib.word_generator import TweetWordGenerator
from src.emoji2label import deepe2l as e2l


TWEETS_TEXT_FILE = '/home/daniel/heMoji/data/hebrew_tweets_3G_mini.tsv'
DATA_PKL_FILE = '/home/daniel/heMoji/data/data.pkl'


def getArgs():
    if len(sys.argv) == 3:
        tweets_text_file = sys.argv[1]
        data_file = sys.argv[2]
    else:
        tweets_text_file = TWEETS_TEXT_FILE
        data_file = DATA_PKL_FILE

    print("""\nLoading tweets text file: "{0}" and Parsing it to data file: "{1}"\n""".format(
        tweets_text_file, data_file))

    return tweets_text_file, data_file


def make(tweets_text_file, data_file):
    with codecs.open(tweets_text_file, 'rU', 'utf-8') as stream:
        wg = TweetWordGenerator(stream, allow_unicode_text=True, wanted_emojis=e2l)
        vb = VocabBuilder(wg)
        vb.set_tweet_tag()
        vb.save_labels(data_file)


if __name__ == '__main__':
    """
    Creates a dataset of (X,Y) from a tsv file.
    Where X is a non parsed sentence (original sentence from the tweet text file)
    Where Y is a label (class number) of some emoji
    """
    tweets_text_file, data_file = getArgs()
    make(tweets_text_file, data_file)
