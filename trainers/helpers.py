from macros import *
import pickle
import numpy as np
import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from trainers.old_helpers import load_subword_embeddings, load_embeddings


def load_meta_embeddings(path):
    """This is assuming that meta-embeddings"""
    return load_vocab()


def get_embeddings(emb_type):
    if 'sub' in emb_type:
        return load_subword_embeddings()
    elif 'meta-pos' in emb_type:
        return load_meta_embeddings("pos")
    else:
        load_embeddings()

def retrieve_embeddings(vocab_terms, emb_type = 'word'):

    pre_embs = get_embeddings(emb_type)
    term_vecs = get_vocab_embeddings(vocab_terms, pre_embs)
    fn = get_fn(emb_type)
    # need to load embedding not model !
    tuned_terms_vecs = load_model(fn)
    tuned_terms_vecs = OrderedDict(
        {term: tuned_terms_vecs.wv[term] for term in
         set(vocab_terms) if term in tuned_terms_vecs.vocab})
    term_vecs = [term_vecs, tuned_terms_vecs]

    return term_vecs, vocab_terms


def get_optimizer(model, args):
    if args.optimizer == 'amsgrad':
        optim = torch.optim.Adam(model.parameters(), amsgrad=True)
    elif args.optimizer == 'sgdr':
        optim = torch.optim.Adam(model.parameters(), amsgrad=True)
        # FINISH !
        # optimizer = torch.optim.CosineA
    elif args.optimizer is not None:
        optim = torch.optim.Adam(model.parameters())
    return optim


def get_scheduler(optim, args, train_data):
    if args.optimizer is not None:
        if args.scheduler == 'cosine_anneal':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20000, eta_min=0, last_epoch=-1)
        elif args.scheduler == 'lro':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
        elif args.scheduler == 'multi_step':
            max_epochs = args.epochs * train_data.size(0) / args.bptt
            mstones = list(range(0, max_epochs, max_epochs / 10))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=mstones, gamma=0.1)
    return scheduler


def load_model(args, model, optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer


def get_fn(vec_name, wiki = 2):
    if vec_name == 'fasttext':
        if wiki == 2:
            return CHUNKING_FASTTEXT_TRAINED_VECTOR_PATH
        else:
            return POS_FASTTEXT_TRAINED_VECTOR_PATH
    elif vec_name == 'word2vec':
        if wiki == 2:
            return POS_WORD2VEC_TRAINED_VECTOR_PATH
        else:
            return POS_WORD2VEC_TRAINED_VECTOR_PATH


# passes a dictionary of attributes created from CHUNKING()
def save_wiki102_vocab(CHUNKING_vocab):
    paths = [CHUNKING_WORD2IND_PATH, CHUNKING_IND2WORD_PATH,
             CHUNKING_WORD2VEC_VOCAB_PATH, CHUNKING_ID2VEC_VOCAB_PATH]
    for path, (name, attrib) in zip(paths, CHUNKING_vocab.items()):
        save_vocab(attrib, path)


def get_vocab_embeddings(word2ind, embs):
    # embs = load_subword_embeddings()
    word2vec = {term: embs[term] if term in embs.vocabulary else np.zeros((300,))
                for term, num in word2ind.items()}
    ind2vec = {ind: word2ind[term] for term, ind in word2ind.items() if term in word2vec}
    return word2vec, ind2vec


def save_vocab(terms, path=CHUNKING_WORD2IND_PATH, show_len=True):
    if show_len:
        print("{} word in vocab.".format(len(terms)))
    with open(path, mode='wb') as pickle_file:
        pickle.dump(terms, pickle_file)


def load_vocab(path = CHUNKING_WORD2IND_PATH):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data


def iterate_paths(paths):
    wiki_vocab = {}
    for path in paths:
        key = path.rsplit('/', 1)[1].replace('.p', '')
        wiki_vocab[key] = load_vocab(path)


def load_wiki103_vocab():
    paths = [POS_NEIGHBOR2VEC_PATH, POS_WORD2IND_PATH,
             POS_IND2WORD_PATH, POS_WORD2VEC_VOCAB_PATH,
             POS_ID2VEC_VOCAB_PATH]
    wiki_vocab = iterate_paths(paths)
    return wiki_vocab


def load_wiki102_vocab():
    paths = [CHUNKING_NEIGHBOR2VEC_PATH, CHUNKING_WORD2IND_PATH,
             CHUNKING_IND2WORD_PATH, CHUNKING_WORD2VEC_VOCAB_PATH,
             CHUNKING_ID2VEC_VOCAB_PATH]
    wiki_vocab = iterate_paths(paths)
    return wiki_vocab


def old_saveandload():
    word2ind = load_vocab()
    id2word = load_vocab(CHUNKING_IND2WORD_PATH)
    word2vec, id2vec = get_vocab_embeddings(word2ind)
    save_vocab(word2vec, CHUNKING_WORD2VEC_VOCAB_PATH)
    save_vocab(id2vec, CHUNKING_ID2VEC_VOCAB_PATH)


def padded_tensor(x, bsz):
    pad_tensor = torch.zeros((x.size(0), bsz - x.size(1))).type(torch.cuda.LongTensor)
    x = Variable(torch.cat([x, pad_tensor], 1))
    return x



