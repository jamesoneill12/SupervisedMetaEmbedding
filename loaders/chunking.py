import re
from trainers.misc import embedding_dict
from torchtext import data
from torchtext.datasets.sequence_tagging import CoNLL2000Chunking
from macros import DATA_PATH
from loaders.dictionary import Dictionary, PretrainedDictionary
import en_core_web_sm
import de_core_news_sm
import torch

spacy_de = en_core_web_sm.load()
spacy_en = de_core_news_sm.load()

url = re.compile('(<url>.*</url>)')


def chunking102(pretrain=False, emb_type='word'):
    corpus = CoNLLCorpus(emb_type='word', pretrain=pretrain)
    corp = {}
    corp['word2ind'] = corpus.dictionary.word2idx
    print("Vocabulary Size: {}".format(len(corp['word2ind'])))
    corp['ind2word'] = corpus.dictionary.idx2word
    corp['word2vec'] = corpus.dictionary.wv
    corp['id2vec'] = None
    return corp


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


class ChunkCorpus:

    def __init__(self, pretrain=False, ns=False, devicer='cuda'):

        # should there be seperate dicts for source and target ?
        self.dictionary = PretrainedDictionary() if pretrain else Dictionary()
        self.devicer = devicer
        self.pretrain = pretrain
        self.ns = ns

        # we need to do this for nsampling unless i'm resampling the input
        # since we can replace the ud tags or ptb tags instead
        if pretrain:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))

        self.BOS_WORD = u'<s>'
        self.EOS_WORD = u'</s>'
        self.BLANK_WORD = u"<blank>"
        self.vocab_field = data.Field(tokenize=tokenize,
                                pad_token=self.BLANK_WORD,
                                init_token=self.BOS_WORD,
                                eos_token=self.EOS_WORD)
        self.vocab = None


class CoNLLCorpus(ChunkCorpus):

    def __init__(self, batch_size, pretrain=False, ns=False, devicer="cuda:0"):

        super(CoNLLCorpus, self).__init__(pretrain, ns, devicer)

        self.devicer = torch.device("cuda:0" if devicer else "cpu")
        self.dictionary = Dictionary()
        self.path = DATA_PATH + "chunking/"
        self.get_data(batch_size)

    def get_data(self, batch_size):
        # Using the CoNLL 2000 Chunking dataset:
        self.inputs = data.Field(init_token=self.BOS_WORD, eos_token=self.EOS_WORD, tokenize='spacy')
        self.tags = data.Field(init_token=self.BOS_WORD, eos_token=self.EOS_WORD)
        train, val, test = CoNLL2000Chunking.splits(root=self.path,
        fields=(('text', self.inputs), (None, None), ('label', self.tags)))

        # train_data = CoNLL2000Chunking(text_field=self.vocab_field, path=self.path+'/all_train.txt')
        self.inputs.build_vocab(train, min_freq=2, max_size=50000)
        self.dictionary.word2idx = dict(self.inputs.vocab.stoi)
        self.dictionary.idx2word = list(self.inputs.vocab.itos)

        self.tags.build_vocab(train.label, min_freq=2, max_size=50000)

        self.dictionary.word2idx = dict(self.inputs.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.tag_vocab = self.tags.vocab.stoi

        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=self.devicer)

        self.train.repeat = False
        self.valid.repeat = False
        self.test.repeat = False

    def __len__(self):
        return len(self.inputs.vocab.itos)


if __name__ == "__main__":
    ccorpus = CoNLLCorpus(batch_size=40)
    print(ccorpus.train.__dict__.keys())

    print("{} number of chunk tags".format(len(ccorpus.tag_vocab)))
    print(ccorpus.tag_vocab)

    cnt = 0
    for i, batch in enumerate(iter(ccorpus.train)):
        cnt += batch.text.size(1)
    print(cnt)



