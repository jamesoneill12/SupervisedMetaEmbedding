# coding: utf-8

import numpy as np
import torch
import math


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

# why not first create batches by grouping ones of similar length then ?
# gets rid of stochasticity but I think it would be better ?

def check_nan(tensor):
    check = torch.isnan(tensor.cpu()).type(torch.ByteTensor).any()
    return check


def check_gradients(model):
    for name, p in model.named_parameters():
        print("{} : Gradient {}".format(name, p.grad is not None))


def get_mt_batches(batch, corpus, joint=False, src='german'):
    """By default German is the source language. """
    if src == 'german':
        source_vocab = corpus.de_vocab
        if joint:
            target_vocab = corpus.en_vocab
    else:
        source_vocab = corpus.en_vocab
        if joint:
            target_vocab = corpus.de_vocab

    x_src = batch.src
    # src_length = batch.src.size(0)
    x_trg = batch.trg[:, :-1]
    x_start = (torch.zeros((x_trg.size(0), 1)) *
               source_vocab[corpus.BOS_WORD]).type(torch.cuda.LongTensor)
    x_trg = torch.cat([x_start, x_trg], dim=1)
    trg_output = batch.trg[:, 1:]
    x_end = (torch.zeros((trg_output.size(0), 1)) *
             source_vocab[corpus.EOS_WORD]).type(torch.cuda.LongTensor)
    trg_output = torch.cat([trg_output, x_end], dim=1)

    if joint:
        """
        Needed when building language model on the source LM. When would we need
        a language model on the source side ? When we want to predict multiple steps
        ahead to create context when only given some words at the beginning.
        """
        x_src_output = batch.src[:, :-1]
        x_src_output_start = (torch.zeros((x_src_output.size(0), 1)) *
                   target_vocab[corpus.BOS_WORD]).type(torch.cuda.LongTensor)
        x_src_output = torch.cat([x_src_output_start, x_src_output], dim=1)
        return x_src, x_trg, trg_output, x_src_output

    return x_src, x_trg, trg_output


def get_mt_batch(x_src, corpus):
    x_start = (torch.zeros((x_src.size(0), 1)) *
               corpus.en_vocab[corpus.BOS_WORD]).type(torch.cuda.LongTensor)
    x_trg = torch.cat([x_start, x_src], dim=1)
    src_output = x_src[:, 1:]
    x_end = (torch.zeros((src_output.size(0), 1)) *
             corpus.en_vocab[corpus.EOS_WORD]).type(torch.cuda.LongTensor)
    trg_output = torch.cat([src_output, x_end], dim=1)
    return x_trg, trg_output


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


