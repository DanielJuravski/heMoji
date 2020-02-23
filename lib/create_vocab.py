from __future__ import print_function, division

import glob
import numpy as np
import uuid
from lib.filter_utils import is_special_token
from lib.word_generator import WordGenerator
from collections import defaultdict, OrderedDict
from lib.global_variables import SPECIAL_TOKENS, VOCAB_PATH
from copy import deepcopy
import json
import pickle


class VocabBuilder():
    """ Create vocabulary with words extracted from sentences as fed from a
        word generator.
    """

    def __init__(self, word_gen):
        # initialize any new key with value of 0
        self.word_counts = defaultdict(lambda: 0, {})
        # self.word_counts = defaultdict()
        self.word_length_limit = 30 * 4
        self.X = []
        self.Y = []

        for token in SPECIAL_TOKENS:
            assert len(token) < self.word_length_limit
            self.word_counts[token] = 0
        self.word_gen = word_gen

    def count_words_in_sentence(self, words):
        """ Generates word counts for all tokens in the given sentence.

        # Arguments:
            words: Tokenized sentence whose words should be counted.
        """
        for word in words:
            word = word.encode('utf-8')
            if 0 < len(word) and len(word) <= self.word_length_limit:
                # word = word.encode('utf-8') # moved up beacuse of un consistent length issues
                # word_u = word
                try:
                    self.word_counts[word] += 1
                    # self.word_counts[word_u] += 1
                # except KeyError:
                except UnicodeEncodeError:
                    self.word_counts[word] = 1
                    # self.word_counts[word_u] = 1

    def save_vocab(self, threshold, path_npz=None, path_json=None):
        """ Saves the vocabulary into 2 files (npz-compressed np file and json file)

        # Arguments:
            path: Where the vocabulary should be saved. If not specified, a
                  randomly generated filename is used instead.
            threshold: if word's count lower than threshold, don't write it to vocab
        """
        dtype = ([('word', '|S{}'.format(self.word_length_limit)), ('count', 'int')])
        np_dict = np.array(self.word_counts.items(), dtype=dtype)

        # sort from highest to lowest frequency
        np_dict[::-1].sort(order='count')
        data = np_dict

        if path_npz is not None:
            np.savez_compressed(path_npz, data=data)
            print("Saved dict to {}".format(path_npz))

        if path_json is None:
            path_json = str(uuid.uuid4())

        with open(path_json, 'w') as f:
            json_d = {}
            i = 0
            dropped_words = 0
            dropped_appears = 0
            for (w, c) in data:
                if c > threshold or w in SPECIAL_TOKENS:
                    w = w.decode('utf-8')#.encode('utf-8')  # for preaty json. if that won't work, remove the encoding.
                    json_d[w] = i
                    i += 1
                else:
                    dropped_words += 1
                    dropped_appears += c
            json.dump(json_d, f, indent=4, sort_keys=True) ## , ensure_ascii=False, encoding=None
            print("Saved dict to {}".format(path_json))
            print("Vocab size is {0}".format(len(json_d)))
            print("Number of dropped words: {0}".format(dropped_words))
            print("Number of dropped appearances: {0}".format(dropped_appears))

    def get_next_word(self):
        """ Returns next tokenized sentence from the word generator.

        # Returns:
            List of strings, representing the next tokenized sentence.
        """
        return self.word_gen.__iter__().next()

    def count_all_words(self):
        """ Generates word counts for all words in all sentences of the word
            generator.
        """
        for words, _, orig_line in self.word_gen:
            self.count_words_in_sentence(words)

    def set_tweet_tag(self):
        """ Generates word's label.
        """
        from src.emoji2label import e2l
        for words, label, orig_line in self.word_gen:
            # sentence = (' '.join(words))

            # make x,y sample for every emoji that appears in the sentence
            # e.g. "sen1_e1_e2_e3" will generate 3 samples (sen1, e1), (sen2, e2), (sen3, e3)
            for e in label['emojis']:
                e_label = e2l[e]
                self.X.append(orig_line.strip('\n'))
                # self.X.append(' '.join(words))
                # self.Y.append({'label': '{0}'.format(e_label)})
                self.Y.append(e_label)  # that is done here/that way/... for using of 'dataset_split' script

    def save_labels(self, path_npz=None, path_json=None):
        """ Saves the labels into 2 files (npz-compressed np file and json file)

        # Arguments:
            path: Where the vocabulary should be saved. If not specified, a
                  randomly generated filename is used instead.
        """

        if path_npz is None:
            path_npz = str(uuid.uuid4())
        if path_json is None:
            path_json = str(uuid.uuid4())

        with open(path_npz, 'wb') as f:
            data = {'X': self.X, 'Y': self.Y}
            pickle.dump(data, f)

        print("Saved labels to {}".format(path_npz))

        # with open(path_json, 'w') as f:
        #     json_d = {}
        #     i = 0
        #     for (w, _) in data:
        #         w = w.decode('utf-8').encode(
        #             'utf-8')  # for preaty json. if that won't work, remove the encoding.
        #         json_d[w] = i
        #         i += 1
        #     json.dump(json_d, f, indent=4, sort_keys=True)
        #     print("Saved dict to {}".format(path_json))


class MasterVocab():
    """ Combines vocabularies.
    """

    def __init__(self):

        # initialize custom tokens
        self.master_vocab = {}

    def populate_master_vocab(self, vocab_path, min_words=1, force_appearance=None):
        """ Populates the master vocabulary using all vocabularies found in the
            given path. Vocabularies should be named *.npz. Expects the
            vocabularies to be numpy arrays with counts. Normalizes the counts
            and combines them.

        # Arguments:
            vocab_path: Path containing vocabularies to be combined.
            min_words: Minimum amount of occurences a word must have in order
                to be included in the master vocabulary.
            force_appearance: Optional vocabulary filename that will be added
                to the master vocabulary no matter what. This vocabulary must
                be present in vocab_path.
        """

        paths = glob.glob(vocab_path + '*.npz')
        sizes = {path: 0 for path in paths}
        dicts = {path: {} for path in paths}

        # set up and get sizes of individual dictionaries
        for path in paths:
            np_data = np.load(path)['data']

            for entry in np_data:
                word, count = entry
                if count < min_words:
                    continue
                if is_special_token(word):
                    continue
                dicts[path][word] = count

            sizes[path] = sum(dicts[path].values())
            print('Overall word count for {} -> {}'.format(path, sizes[path]))
            print('Overall word number for {} -> {}'.format(path, len(dicts[path])))

        vocab_of_max_size = max(sizes, key=sizes.get)
        max_size = sizes[vocab_of_max_size]
        print('Min: {}, {}, {}'.format(sizes, vocab_of_max_size, max_size))

        # can force one vocabulary to always be present
        if force_appearance is not None:
            force_appearance_path = [p for p in paths if force_appearance in p][0]
            force_appearance_vocab = deepcopy(dicts[force_appearance_path])
            print(force_appearance_path)
        else:
            force_appearance_path, force_appearance_vocab = None, None

        # normalize word counts before inserting into master dict
        for path in paths:
            normalization_factor = max_size / sizes[path]
            print('Norm factor for path {} -> {}'.format(path, normalization_factor))

            for word in dicts[path]:
                if is_special_token(word):
                    print("SPECIAL - ", word)
                    continue
                normalized_count = dicts[path][word] * normalization_factor

                # can force one vocabulary to always be present
                if force_appearance_vocab is not None:
                    try:
                        force_word_count = force_appearance_vocab[word]
                    except KeyError:
                        continue
                    # if force_word_count < 5:
                        # continue

                if word in self.master_vocab:
                    self.master_vocab[word] += normalized_count
                else:
                    self.master_vocab[word] = normalized_count

        print('Size of master_dict {}'.format(len(self.master_vocab)))
        print("Hashes for master dict: {}".format(
            len([w for w in self.master_vocab if '#' in w[0]])))

    def save_vocab_npz(self, path_count, path_vocab, word_limit=100000):
        """ Saves the master vocabulary into a file.
        """

        # reserve space for 10 special tokens
        words = OrderedDict()
        for token in SPECIAL_TOKENS:
            # store -1 instead of np.inf, which can overflow
            words[token] = -1

        # sort words by frequency
        desc_order = OrderedDict(sorted(self.master_vocab.items(),
                                        key=lambda kv: kv[1], reverse=True))
        words.update(desc_order)

        # use encoding of up to 30 characters (no token conversions)
        # use float to store large numbers (we don't care about precision loss)
        np_vocab = np.array(words.items(),
                            dtype=([('word', '|S30'), ('count', 'float')]))

        # output count for debugging
        counts = np_vocab[:word_limit]
        np.savez_compressed(path_count, counts=counts)

        # output the index of each word for easy lookup
        final_words = OrderedDict()
        for i, w in enumerate(words.keys()[:word_limit]):
            final_words.update({w: i})
        with open(path_vocab, 'w') as f:
            f.write(json.dumps(final_words, indent=4, separators=(',', ': ')))


def all_words_in_sentences(sentences):
    """ Extracts all unique words from a given list of sentences.

    # Arguments:
        sentences: List or word generator of sentences to be processed.

    # Returns:
        List of all unique words contained in the given sentences.
    """
    vocab = []
    if isinstance(sentences, WordGenerator):
        sentences = [s for s, _ in sentences]

    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)

    return vocab


def extend_vocab_in_file(vocab, max_tokens=10000, vocab_path=VOCAB_PATH):
    """ Extends JSON-formatted vocabulary with words from vocab that are not
        present in the current vocabulary. Adds up to max_tokens words.
        Overwrites file in vocab_path.

    # Arguments:
        new_vocab: Vocabulary to be added. MUST have word_counts populated, i.e.
            must have run count_all_words() previously.
        max_tokens: Maximum number of words to be added.
        vocab_path: Path to the vocabulary json which is to be extended.
    """
    try:
        with open(vocab_path, 'r') as f:
            current_vocab = json.load(f)
    except IOError:
        print('Vocabulary file not found, expected at ' + vocab_path)
        return

    extend_vocab(current_vocab, vocab, max_tokens)

    # Save back to file
    with open(vocab_path, 'w') as f:
        json.dump(current_vocab, f, sort_keys=True, indent=4, separators=(',', ': '))


def extend_vocab(current_vocab, new_vocab, max_tokens=10000):
    """ Extends current vocabulary with words from vocab that are not
        present in the current vocabulary. Adds up to max_tokens words.

    # Arguments:
        current_vocab: Current dictionary of tokens.
        new_vocab: Vocabulary to be added. MUST have word_counts populated, i.e.
            must have run count_all_words() previously.
        max_tokens: Maximum number of words to be added.

    # Returns:
        How many new tokens have been added.
    """
    if max_tokens < 0:
        max_tokens = 10000

    words = OrderedDict()

    # sort words by frequency
    desc_order = OrderedDict(sorted(new_vocab.word_counts.items(),
                                    key=lambda kv: kv[1], reverse=True))
    words.update(desc_order)

    base_index = len(current_vocab.keys())
    added = 0
    for word in words:
        if added >= max_tokens:
            break
        if word not in current_vocab.keys():
            current_vocab[word] = base_index + added
            added += 1

    return added
