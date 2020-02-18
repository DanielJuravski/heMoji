""" Creates a vocabulary from a tsv file.
"""

import codecs
from lib.create_vocab import VocabBuilder
from lib.word_generator import TweetWordGenerator
import sys
from src.emoji2label import e2l

TWEETS_TEXT_FILE = '/home/daniel/heMoji/data/hebrew_tweets_3G_mini.tsv'
VOCAB_FILE = '/home/daniel/heMoji/data/vocabulary.json'


def getArgs():
    if len(sys.argv) == 3:
        tweets_text_file = sys.argv[1]
        vocab_file = sys.argv[2]
    else:
        tweets_text_file = TWEETS_TEXT_FILE
        vocab_file = VOCAB_FILE

    print("""\nLoading tweets text file: "{0}" and Parsing it to vocab file: "{1}"\n""".format(
        tweets_text_file, vocab_file))

    return tweets_text_file, vocab_file


def make(tweets_text_file, vocab_file):
    with codecs.open(tweets_text_file, 'rU', 'utf-8') as stream:
        wg = TweetWordGenerator(stream, allow_unicode_text=True, wanted_emojis=e2l)
        vb = VocabBuilder(wg)
        vb.count_all_words()
        vb.save_vocab(path_json=vocab_file)
        pass


if __name__ == '__main__':
    """
    Usage: python make_vocab.py [TWEETS_TEXT_FILE] [VOCAB_FILE]
    creating vocab at the form of ["word": token_number]
    """
    tweets_text_file, vocab_file = getArgs()
    make(tweets_text_file, vocab_file)
