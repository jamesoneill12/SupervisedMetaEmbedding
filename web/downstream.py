# -*- coding: utf-8 -*-


import spacy
from torchtext import datasets, data
import torchtext
import random
import torch

ROOT_PATH = "C:/Users/jimon/Projects/word-embeddings-benchmarks/web_data/"
DOWNSTREAM_PATH = ROOT_PATH+ 'downstream'
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

"""
Fetch dataset for testing attributional similarity

Returns
-------
data : sklearn.datasets.base.Bunch
    dictionary-like object. Keys of interest:
    'X': matrix of 2 words per column,
    'y': vector with scores,
"""

spacy_en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tabular_iter(text, label):
    train, val, test = torchtext.data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('Text', text), ('Label', label)])
    data = {'train': train, 'val': val, 'test': test }
    return (data)


def tt_bucket_iter(train, test):
    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=3, device=0)
    return (train_iter, test_iter)


def flatten(l): return ([item for sublist in l for item in sublist])


class Downstream():
    # vocab set to True, we just deal with returning vocab, otherwise the train, test, val splits
    def __init__(self):
        self.TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        self.LABEL = data.Field(sequential=False)

    # below two functions for when you want to pass in your own data
    def get_data(self, train):
        # build the vocabulary
        self.TEXT.build_vocab(train, max_size=30000)
        # vectors=GloVe(name='6B', dim=300))
        self.LABEL.build_vocab(train)
        return (self.TEXT, self.LABEL)

    def get_iters(self, train, test):
        train_iter, test_iter = tt_bucket_iter(train, test)
        data = {'train': train_iter, 'test': test_iter}
        return (data)

    def fetch_SENT_vocab(self):
        try:
            vocab = self.sent_train + self.sent_valid +self.sent_test
        except:
            train, self.sent_test = datasets.IMDB.splits(self.TEXT, self.LABEL)
            self.sent_train, self.sent_valid = train.split(random_state=random.seed(SEED))
            self.TEXT.build_vocab(self.pos_train.word, min_freq=3)
            vocab = self.TEXT.vocab.itos
        return vocab

    def fetch_SENT(self):
        train, self.sent_test = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.sent_train, self.sent_valid = train.split(random_state=random.seed(SEED))
        device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (self.sent_train, self.sent_valid, self.sent_test), batch_size= 20, device=device)
        return (train_iter, val_iter, test_iter)

    def fetch_POS_vocab(self):
        try:
            vocab = self.pos_train + self.pos_valid + self.pos_test
        except:
            # Define the fields associated with the sequences.
            WORD = data.Field(init_token="<bos>", eos_token="<eos>")
            UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
            PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
            self.pos_train, self.pos_valid, self.pos_test = \
                datasets.UDPOS.splits(
                    fields=(('word', WORD), ('udtag', UD_TAG), ('ptbtag', PTB_TAG))
                )
            WORD.build_vocab(self.pos_train.word, min_freq=3)

            UD_TAG.build_vocab(self.pos_train.udtag)
            PTB_TAG.build_vocab(self.pos_train.ptbtag)
            vocab = WORD.vocab.itos
        return vocab

    def fetch_POS(self):
        train, self.pos_test =  datasets.UDPOS.splits(self.TEXT, self.LABEL)
        self.pos_train, self.pos_valid = train.split(random_state=random.seed(SEED))
        device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (self.pos_train, self.pos_valid, self.pos_test), batch_size= 20, device=device)

        print("Batch Info")
        batch = next(iter(train_iter))
        print(batch.text)
        print(batch.label)

        return train_iter, val_iter, test_iter

    def fetch_TREC_vocab(self):
        try:
            vocab = self.get_vocab(self.trec_train, self.trec_test)
        except:
            train, self.trec_test = datasets.TREC.splits(self.TEXT, self.LABEL, fine_grained=True)
            self.trec_train, self.trec_valid = train.split(random_state=random.seed(SEED))
            self.TEXT.build_vocab(self.trec_train.word, min_freq=3)
            vocab = self.TEXT.vocab.itos
        return vocab

    def fetch_TREC(self):
        train, self.trec_test =  datasets.TREC.splits(self.TEXT, self.LABEL, fine_grained=True)
        self.trec_train, self.trec_valid = train.split(random_state=random.seed(SEED))
        device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (self.trec_train, self.trec_valid, self.trec_test), batch_size= 20, device=device)
        return train_iter, val_iter, test_iter


