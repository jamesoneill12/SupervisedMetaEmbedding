import torch.utils.data
import torch
from loaders.dictionary import PoSDictionary
from torchtext import data
from torchtext import datasets
from trainers.misc import embedding_dict
from macros import DATA_PATH


class PoSCorpus:
    def __init__(self, batch_size=32, ud=True, pretrain=False,
                 ns=False, reward=False, devicer='cpu'):

        self.dictionary = PoSDictionary()
        self.devicer = torch.device("cuda:0" if devicer else "cpu")
        self.path = DATA_PATH + "udpos/"
        self.batch_size = batch_size
        self.ud = ud

        # we need to do this for nsampling unless i'm resampling the input
        # since we can replace the ud tags or ptb tags instead
        if ns or reward and pretrain is False:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))

        self.get_data()

    def get_data(self):
        # Define the fields associated with the sequences.
        self.inputs = data.Field(init_token="<bos>", eos_token="<eos>", tokenize='spacy')
        self.UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
        self.PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

        # Download and the load default data.
        print(self.path)
        # was udtag
        train_data, val_data, test_data = datasets.UDPOS.splits(root=self.path,
            fields=(('text', self.inputs), ('label', self.UD_TAG), ('ptbtag', self.PTB_TAG))
                                                                )

        self.inputs.build_vocab(train_data, min_freq=3)
        self.dictionary.word2idx = dict(self.inputs.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())

        self.UD_TAG.build_vocab(train_data.label)
        self.PTB_TAG.build_vocab(train_data.ptbtag)

        if self.ud:
            self.tag_vocab = self.UD_TAG.vocab.stoi
            self.ptb_vocab = self.PTB_TAG.vocab.stoi
        else:
            self.ud_vocab = self.UD_TAG.vocab.stoi
            self.tag_vocab = self.PTB_TAG.vocab.stoi

        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size=self.batch_size, device=self.devicer)

        self.train.repeat = False
        self.valid.repeat = False
        self.test.repeat = False

    def __len__(self):
        return len(self.inputs.vocab.itos)


if __name__ == "__main__":
    # remember this can work as a mtl problem where learn to predict the udtag and postag
    pc = PoSCorpus()
    print(pc.dictionary.word2idx.keys())
    cnt = 0
    for i, (batch) in enumerate(iter(pc.train)):
        assert batch.text.size(0) == len(batch.label)
        cnt+=batch.text.size(1)

    print(cnt)

