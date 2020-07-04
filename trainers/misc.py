import numpy as np
import re
from trainers.helpers import get_embeddings


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def inc_join(sentences, splits=10):
    """
    use for when out of memory must incremtally go through
    """
    fs_len = len(sentences)
    split_range = int(fs_len / float(splits))
    vocab_terms = []
    for i, batch in enumerate(range(0, fs_len, split_range)):
        if i == 0:
            sent_batch = sentences[0:batch]
        else:
            sent_batch = sentences[batch - split_range:batch]
        vocab_terms += list(set(clean_str(" ".join(sent_batch)).lower().split()))
    return set(vocab_terms)


def get_uniq_tokens(sentences, inc=True):
    if inc:
        return inc_join(sentences)
    return set(" ".join(sentences).lower().split())


def get_vocab(data, incremental=True):
    flattened_sentences = list(flatten(data))
    corpus_string = get_uniq_tokens(flattened_sentences, inc=incremental)

    # just clean unique faster and less memory
    tokens = data.clean_str(" ".join(list(corpus_string)))
    vocab = list(set(tokens.split()))
    word2ind = {word: i for i, word in enumerate(vocab)}
    ind2word = {v: k for k, v in word2ind.items()}
    return word2ind, ind2word


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# MAKE SURE id: NOT word: IN SAVED SERVER FILE
def embedding_dict(word2idx, emb_type='word'):
    embedding = get_embeddings(emb_type)
    id2vec = {id : (embedding.wv[word] if word in embedding else np.zeros((300,))) for (word, id) in word2idx.items()}
    return id2vec


def normalize(x, axis=0):
    return(x-np.min(x, axis))/(np.max(x, axis)-np.min(x, axis))


def softmax(x, axis=0, temp=1.0):

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis))/temp
    return e_x / e_x.sum(axis=axis)  # only difference
