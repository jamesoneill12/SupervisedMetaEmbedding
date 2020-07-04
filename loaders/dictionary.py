from trainers.helpers import get_embeddings, get_vocab_embeddings
from trainers.misc import clean_str
import nltk
from collections import Counter
from edlib import align
import numpy as np


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class PoSDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


class NERDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.limited_vocab = []

    def clean_word(self, word):
        return clean_str(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def limited_word(self, word):
        if word not in self.limited_vocab:
            self.limited_vocab.append(word)

    def remove_word(self, word):
        if word in self.word2idx:
            i = self.idx2word.index(word)
            del self.idx2word[i]
            self.idx2word.remove(word)
            self.word2idx.pop(word, None)

    """idx limit ensures that only the indices from inside the vocab are checked"""
    def replace_word(self, word, normalize=True):
        """uses levenshtein by default, no argument to avoid time spent on conditionals"""
        """edlib.align faster than levenshteinDistance"""
        if normalize:
            scores = []
            for vocab_word in self.limited_vocab:
                align_score = align(vocab_word, word)
                edit_distance = align_score['editDistance']
                if edit_distance != 0:
                    edit_distance /= align_score['alphabetLength']
                scores.append(edit_distance)
            scores = np.array([scores])
        else:
            scores = np.array([align(vocab_word, word)['editDistance'] for vocab_word in self.limited_vocab])
        replacement_word = self.idx2word[np.argmin(scores)]
        self.word2idx[word] = self.word2idx[replacement_word]

    def __len__(self):
        # was idx2word but changed for limword2idx assignment
        # changed to set since when limit is set most are given <unk>
        return len(set(list(self.word2idx.values())))


# this should map inds to pretrain inds
class PretrainedDictionary(object):
    def __init__(self, emb_type='word', nn=False):
        self.nn = nn
        self.word2idx = {}
        self.idx2word = []
        self.wv = {}
        self.embeddings = get_embeddings(emb_type)
        self.vocab_perc = 0

        if self.nn:
            self.id2neighbor = {}
            # word id -> neighbor id
            self.wid2nid = {}
            # I also need a map between word id and neighbor id because when setting nn.Embedding we
            # have not included neighor embeddings. This prob involves changing get_neighbors to return

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            id = len(self.idx2word) - 1
            self.word2idx[word] = id
            if word in self.embeddings:
                self.wv[id] = self.embeddings.wv[word]
                # cannot add neighbor ids while building vocab unless I am planning to
                #add all neighbors that are not necessarily words within the corpus.
                #if self.nn:
                #    # returns matrix with neighbor vector and sample probabilities
                #    self.id2neighbor[id] = get_nearest_neighbor(word, self.embeddings)
            else:
                self.vocab_perc +=1
                self.wv[id] = np.zeros((300,))
        return self.word2idx[word]

    def vocab_embeds(self, vocab_terms = None):
        if vocab_terms == None:
            vocab_terms = list(self.word2idx.keys())
        self.wv, self.word2id, self.id2word = get_vocab_embeddings(vocab_terms, self.wv)

    def __len__(self):
        return len(self.idx2word)
