from macros import *
from gensim.models import KeyedVectors
from collections import OrderedDict


def load_embeddings():
    google_model = KeyedVectors.load_word2vec_format(GOOGLE_VECTOR_PATH, binary=True)
    return google_model


def load_subword_embeddings(normalize=True, lower=True, clean_words=True):
    fasttext_model = KeyedVectors.load_word2vec_format(FASTTEXT_VECTOR_PATH)
    return fasttext_model



def get_vocab_embeddings_old(vocab_terms, pre_embs):
    print("Original vocabulary length : {}".format(len(vocab_terms)))
    word2vec = OrderedDict(
        {term: pre_embs.wv[term] for term in set(vocab_terms) if term in pre_embs.vocab})
    percentage_retrieved = 100 * len(word2vec) / float(len(vocab_terms))
    print("{} term vectors are available without spellchecking".format(percentage_retrieved))
    word2id = {word: i for i, word in enumerate(word2vec.keys())}
    id2word = {i: word for i, word in enumerate(word2vec.items())}
    return word2vec, word2id, id2word

