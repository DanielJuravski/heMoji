import json
from collections import defaultdict, OrderedDict

from raw_to_pickle import load_tsv_data

VOCAB_PATH = '/home/daniel/heMoji/data/vocab_500G_rare5_data01.json'


def load_vocab():
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    return vocab


def iterate_words(data, vocab):
    """
    iterate sentences, iterate words,
    add every word to d_all_words,
    add word that not in the input vocab to d_oov_words
    :param data: list of sentences (strings)
    :param vocab: json, dict ov key (word) - value (word's id)
    :return:
    """
    def sort_dict_vals(d):
        # sort the dict by the highest values
        return OrderedDict(sorted(d.iteritems(), key=lambda (k, v): v, reverse=True))

    d_all_words = defaultdict(int)
    d_oov_words = defaultdict(int)
    for sentence in data:
        words = sentence.split()
        for word in words:
            word_u = word.decode('utf-8')
            if word_u not in vocab:
                d_oov_words[word] += 1
            d_all_words[word] += 1

    d_all_words = sort_dict_vals(d_all_words)
    d_oov_words = sort_dict_vals(d_oov_words)

    return d_all_words, d_oov_words


def dump_stats(d, file_suffix):
    with open(TYPE + file_suffix, 'w') as f:
        f.writelines("words_count:\t{0}\n".format(sum([v for k,v in d.iteritems()])))
        f.writelines("tokens_count:\t{0}\n".format(len(d)))
        f.writelines("vocab:\t{0}\n".format(VOCAB_PATH))
        f.writelines("\n")
        for k,v in d.iteritems():
            f.writelines("{0}\t{1}\n".format(k, v))


if __name__ == '__main__':
    TYPE = 'token'
    train_X, train_Y, test_X, test_Y = load_tsv_data(TYPE)
    vocab = load_vocab()

    d_all_words_train, d_oov_words_train = iterate_words(train_X, vocab)
    d_all_words_test, d_oov_words_test = iterate_words(test_X, vocab)

    dump_stats(d_all_words_train, '_train_all_words.txt')
    dump_stats(d_oov_words_train, '_train_oov_words.txt')

