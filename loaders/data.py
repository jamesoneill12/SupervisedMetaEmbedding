import os
import torch
import numpy as np
from trainers.misc import embedding_dict, clean_str
from macros import *
from loaders.dictionary import Dictionary, PretrainedDictionary
import time


class Corpus(object):
    def __init__(self, path, emb_type=None, clean=False, limit=None,
                 pretrain=False):

        self.dictionary = Dictionary()
        self.limit = limit

        endext = '.tokens' if '3' in path else '.txt'
        startext = 'wiki.' if '3' in path else ''

        train_path = os.path.join(path, startext + 'train' + endext)
        self.valid_path = os.path.join(path, startext + 'valid' + endext)
        self.test_path = os.path.join(path, startext + 'test' + endext)

        tokenizer = self.tokenize if endext == '.tokens' else self.tokenize

        print("Limit is {}".format(limit))
        if self.limit:
            self.limit_vocab_created = False
            if type(self.limit) != int:
                limit = int(self.limit)
            self.token_count = {}

        self.train = tokenizer(train_path, limit=self.limit, clean=clean)

        if self.limit:
            self.limit_vocab_created = True

        self.valid = tokenizer(self.valid_path)
        self.test = tokenizer(self.test_path)

        if self.limit:
            """needed so that when getting ntokens from __len__"""
            # self.dictionary.word2idx = self.limword2idx
            print("Size of limited vocab {}".format(len(self.dictionary)))

        # not needed after processing
        if hasattr(self.dictionary, 'embeddings'):
            del self.dictionary.embeddings

        # we need to do this for nsampling, sampling neighbors at training time is not reliable.
        if pretrain is False:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))

    def get_limited_vocab(self, token_count, limit):
        # just delete words which have a count > limit
        counts = np.array(list(token_count.values()), dtype=int)
        words = np.array(list(token_count.keys()))
        rare_count_inds = np.argsort(counts)[::-1][limit:]
        common_count_inds = np.argsort(counts)[::-1][:limit]

        limited_vocab = words[common_count_inds]
        oov_words = list(words[rare_count_inds])
        limit_vocab_perc = (len(oov_words) / len(token_count)) * 100

        print("Limited vocab is size {}".format(len(limited_vocab)))
        print("OOV vocab is size {}".format(len(oov_words)))

        print("{:.2f}% of OOV words for replacement".format(limit_vocab_perc))
        if self.lcs:
            print("now replacing rare words using"
                  " closest longest common subsequence ...")
        else:
            print("now replacing rare words using <unk> token")

        """creates a limited vocabulary that is needed to assign rare idx in next step."""
        for lim_word in list(limited_vocab):
            self.dictionary.limited_word(lim_word)

        return oov_words, limited_vocab

    def lcs_vocab_limit(self, token_count, limit):
        oov_words, _ = self.get_limited_vocab(token_count, limit)
        start = time.time()
        for i, rword in enumerate(oov_words):
            """better to replace word than remove so ids can be assigned down below"""
            # self.dictionary.remove_word(rword)
            if i % 100 == 0:
                end = time.time()
                print("{} words processed for replacement, it took {}secs".format(i, start - end))
                start = time.time()
            self.dictionary.replace_word(rword, normalize=True)
        print("replacement finished ! it took {} seconds".format(end - start))

    def tokenize_file(self, path, tokens, clean):
        # Tokenize file content
        with open(path, mode='r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if clean:
                    line = clean_str(line)
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def tokenize_limit_file(self, path, tokens, limited_vocab=None):
        """
        When
        :param path:
        :param tokens:
        :param limited_vocab:
        :return:
        """

        # Tokenize file content and assign id from limited vocab
        if self.limit_vocab_created is False:
            # need to convert all original indices to limited idx
            self.idx2limidx = {self.dictionary.word2idx[word]: i for (i, word) in enumerate(limited_vocab)}
            temp_dict = {}
            for word in list(self.dictionary.word2idx.keys()):
                if word in limited_vocab:
                    temp_dict[word] = self.idx2limidx[self.dictionary.word2idx[word]]
                else:
                    temp_dict[word] = self.dictionary.word2idx['<unk>']

            self.dictionary.word2idx = temp_dict
            del temp_dict

            """
            self.limword2idx = {word : self.idx2limidx[self.dictionary.word2idx[word]]
                        for word in limited_vocab}
            print("Limited vocab is of size {}".format(len(self.limword2idx)))
            """

        with open(path, mode='r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                    """
                    if word in self.limword2idx:
                        ids[token] = self.limword2idx[word]
                    else:
                        ids[token] = self.limword2idx['<unk>']
                    token += 1                    

                    """
        return ids

    def tokenize(self, path, limit=None,
                 clean=False, lcs=False, count=False):

        """
        Tokenizes a text file

        :param
        clean: cleans string of any unneeded punctuation
        limit: when not None and LCS is False,
               just assign oov words to <unk>.
        lcs: when true and limit not None, we assign oov
                words to nearest in-vocab word according to minimum lcs distance (this is slow atm)
        """

        assert os.path.exists(path)
        # Add words to the dictionary

        with open(path, mode='r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                if clean:
                    line = clean_str(line)
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if limit is not None:
                        if word in self.token_count:
                            self.token_count[word] += 1
                        else:
                            self.token_count[word] = 1
                    self.dictionary.add_word(word)

        if limit is not None:
            """  has to be next condition because token_count
            doesn't exist unless limit is not None """
            if len(self.token_count) > limit:
                if lcs:
                    self.lcs_vocab_limit(self.token_count, limit)
                    ids = self.tokenize_file(path, tokens, clean)
                else:
                    _, limited_vocab = self.get_limited_vocab(self.token_count, limit)
                    print("Size of limite")
                    ids = self.tokenize_limit_file(path, tokens, limited_vocab)
            else:
                ids = self.tokenize_file(path, tokens, clean)

        elif self.limit:

            print("Now computing valid and test")
            # no need for limited vocab now since limword2id already created
            ids = self.tokenize_limit_file(path, tokens)

            if 'val' in path: self.val_ids = ids
            elif 'test' in path: self.test_ids = ids

        else:
            ids = self.tokenize_file(path, tokens, clean)

        if hasattr(self.dictionary, 'vocab_perc'):
            vocab_perc = round((1 - self.dictionary.vocab_perc / len(ids)) * 100, 2)
            print("{} % in the pretrained vocab.".format(vocab_perc))

        """get counts from corpus if negative sampling is true"""
        if count: self.id_counts = ids

        return ids


if __name__ == "__main__":
    pass
    # sentences = wikitext103()
    # vocab = get_vocab(sentences)


