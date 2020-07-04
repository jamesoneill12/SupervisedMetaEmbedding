from torchtext import datasets
import torch
from torchtext import data
from loaders.dictionary import Dictionary
from trainers.misc import embedding_dict
from macros import DATA_PATH
import random
from macros import SENT_ROOT

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

class SentimentCorpus:

    def __init__(self,  batch_size, pretrain=False):
        self.dictionary = Dictionary()
        self.path = DATA_PATH + "sentiment/imdb"
        self.devicer = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if pretrain:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))
        self.get_data(batch_size)

    def get_data(self, batch_size):

        self.inputs = data.Field(tokenize='spacy') #, init_token="<bos>", eos_token="<eos>")
        self.tags = data.Field()
        train_data, test_data = datasets.IMDB.splits(self.inputs, self.tags, root=self.path)
        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of testing examples: {len(test_data)}')

        # train_data, valid_data = train_data.split(random_state=random.seed(SEED))

        self.inputs.build_vocab(train_data, max_size=25000)
        self.tags.build_vocab(train_data.label)

        self.dictionary.word2idx = dict(self.inputs.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.tag_vocab = self.tags.vocab.stoi

        #  valid_data, self.valid,
        self.train, self.test = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=batch_size,
            device=self.devicer)

        self.train.repeat = False
        self.test.repeat = False

    def __len__(self):
        return len(self.inputs.vocab.itos)


if __name__ == "__main__":

    ccorpus = SentimentCorpus(batch_size=1000)
    print(ccorpus.train.__dict__.keys())
    print("{} number of chunk tags".format(len(ccorpus.tag_vocab)))
    print(ccorpus.tag_vocab)

    cnt = 0
    for i, batch in enumerate(ccorpus.test):
        cnt += batch.text.size(1)

    print(cnt)