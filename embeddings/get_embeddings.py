from loaders.sentiment import SentimentCorpus
from loaders.chunking import CoNLLCorpus
from loaders.pos import PoSCorpus
from loaders.ner import NERCorpus
from loaders.embeddings import load_embedding
from trainers.helpers import get_vocab_embeddings, save_vocab, load_vocab
import os.path
import numpy as np
import torch
from torch import nn

vec_names = ["lexvec", "glove", "skipgram", "numberbatch", "hdc", "fasttext"]
formats = ['word2vec', 'glove', 'word2vec_bin', 'glove', "glove", "word2vec"]
embed_names = ["crawl-300d-2M.vec", "glove.840B.300d.txt",
               "GoogleNews-vectors-negative300.bin", "numberbatch-en.txt",
               "wikicorp.201004-hdc-iter-20-alpha-0.025-window-10-dim-300-neg-10-subsample-0.0001.txt",
               "wiki-news-300d-1M-subword.vec"]

def get_task(which, batch_size = 32):
    if which == "all":
        return [SentimentCorpus(batch_size), CoNLLCorpus(batch_size), PoSCorpus(batch_size),
                NERCorpus(batch_size)], ["sentiment", "chunking", "pos", "ner"]
    elif which == "sentiment":
        return [SentimentCorpus(batch_size)], ["sentiment"]
    elif which == "chunking":
        return [CoNLLCorpus(batch_size)], ["chunking"]
    elif which == "pos":
        return [PoSCorpus(batch_size)], ["pos"]
    elif which == "ner":
        return [NERCorpus(batch_size)], ["ner"]


def get_vocab(which="all"):
    tasks, names = get_task(which)
    task_vocabs = {name : task.dictionary.word2idx for task, name in zip(tasks, names)}
    return task_vocabs


def get_embeddings(emb_path, format, vec_name):
    # if os.path.exists(save_path)
    if vec_name == "glove" or vec_name == "numberbatch" or vec_name == "hdc":
        if vec_name == "glove":
            vocab_size = int(2.3e6)
        elif vec_name == "numberbatch":
            vocab_size = 1984681
        elif vec_name == "hdc":
            vocab_size = 201004
        pre_embs = load_embedding(emb_path, format=format,
                                  load_kwargs={'dim': 300, 'vocab_size': vocab_size})
    else:
        pre_embs = load_embedding(emb_path, format=format)

    return pre_embs


def save_vocab_embeddings(which="all", root = "embeddings/"):
    vocabs = get_vocab(which)
    embed_paths = {vec_name: root+embed_fname for vec_name, embed_fname in zip(vec_names, embed_names)}

    for i, (vec_name, emb_path) in enumerate(embed_paths.items()):
        pre_embs = get_embeddings(emb_path, formats[i], vec_name)
        for task, w2ind_vocab in vocabs.items():
            save_name = vec_name + "_" + task
            save_path = "{}/{}.pkl".format(task, save_name)
            if os.path.exists(save_path) is False:
                print("saving {} for task {}".format(vec_name, task))
                word2vec, id2word = get_vocab_embeddings(w2ind_vocab, pre_embs)
                save_emb_obj = {'word2vec': word2vec, "id2word": id2word}
                save_vocab(save_emb_obj, path=save_path, show_len=True)
            else:
                print("{} already saved..".format(save_name))


def choose_emb(which):
    if which == "all":
        vec_names = ["lexvec", "glove", "skipgram", "numberbatch", "hdc", "fasttext"]
    elif which == "lexvec":
        vec_names = ["lexvec"]
    elif which == "glove":
        vec_names = ["glove"]
    elif which == "skipgram":
        vec_names = ["skipgram"]
    elif which == "numberbatch":
        vec_names = ["numberbatch"]
    elif which == "hdc":
        vec_names = ["hdc"]
    elif which == "fasttext":
        vec_names = ["fasttext"]
    return vec_names


def load_vocab_embeddings(which="all"):
    vec_names = choose_emb(which)
    tasks = ["sentiment", "ner", "chunking", "pos"]
    for task in tasks:
        for vec_name in vec_names:
            save_name = vec_name + "_" + task
            save_path = "{}/{}.pkl".format(task, save_name)
            emb = load_vocab(path=save_path)
            print(vec_name)
            print(emb["word2vec"])
        break


def load_vocab_embedding(vec_name, task, root = "embeddings"):
    save_name = vec_name + "_" + task
    save_path = "{}/{}/{}.pkl".format(root, task, save_name)
    emb = load_vocab(path=save_path)
    return emb


# chunking and sentiment is ok
def load_meta_embedding(task="chunking", emb_choice="all", tune=False):

    vec_names = choose_emb(emb_choice)
    if len(vec_names) > 1:
        w2v = []
        for vec_name in vec_names:
            emb = load_vocab_embedding(vec_name, task)
            word_vecs = np.array(list(emb['word2vec'].values()))
            w2v.append(word_vecs)
        w2v = torch.from_numpy(np.hstack(w2v))
    else:
        emb = load_vocab_embedding(vec_names[0], task)
        word_vecs = np.array(list(emb['word2vec'].values()))
        w2v = torch.from_numpy(word_vecs)

    embedding = nn.Embedding(w2v.size(0), w2v.size(1))
    embedding.weight.data.copy_(w2v)
    embedding.weight.requires_grad = tune
    return embedding


if __name__ == "__main__":

    save_vocab_embeddings("pos")
    # embedding = load_meta_embedding("ner")
    # print(embedding.weight.data.size())