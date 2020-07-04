import pickle
from .embedding import Embedding

def load_embedding(fname, format="word2vec_bin", normalize=True,
                   lower=False, clean_words=False, load_kwargs={}):
    """
    Loads embeddings from file

    Parameters
    ----------
    fname: string
      Path to file containing embedding

    format: string
      Format of the embedding. Possible values are:
      'word2vec_bin', 'word2vec', 'glove', 'dict'

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.
    """
    assert format in ['word2vec_bin', 'word2vec', 'glove', 'dict'], "Unrecognized format"
    if format == "word2vec_bin":
        w = Embedding.from_word2vec(fname, binary=True)
    elif format == "word2vec":
        w = Embedding.from_word2vec(fname, binary=False)
    elif format == "glove":
        w = Embedding.from_glove(fname, **load_kwargs)
    elif format == "dict":
        d = pickle.load(open(fname, "rb"), encoding='latin1')
        w = Embedding.from_dict(d)
    if normalize:
        w.normalize_words(inplace=True)
    if lower or clean_words:
        w.standardize_words(lower=lower, clean_words=clean_words, inplace=True)
    return w