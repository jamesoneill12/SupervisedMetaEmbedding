import torch.utils.data
import torch
from loaders.dictionary import NERDictionary
from torchtext import datasets
from trainers.misc import embedding_dict
from macros import DATA_PATH
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vectors, GloVe, CharNGram
import logging


class NERCorpus:
    def __init__(self, batch_size, ner_type='nyt_ingredients_ner', pretrain=False, devicer='cpu'):

        self.dictionary = NERDictionary()
        self.devicer = torch.device("cuda" if devicer else "cpu")
        self.path = DATA_PATH + "ner/" #nyt_ingredients_ner/"
        self.ner_type = ner_type

        if pretrain:
            self.dictionary.wv = embedding_dict(self.dictionary.word2idx)
            print("{} words in vocab".format(len(self.dictionary.word2idx)))

        self.get_data(batch_size)

    def get_data(self, batch_size, convert_digits=True):

        # Setup fields with batch dimension first
        inputs = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, tokenize='spacy',
                                 preprocessing=data.Pipeline(
                                     lambda w: '0' if convert_digits and w.isdigit() else w))
        tags = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
        fields = (('text', inputs), ('label', tags))
        # Download and the load default data.
        # train, val, test = datasets.sequence_tagging.SequenceTaggingDataset.splits(root=self.path, fields=fields)
        train, val, test = Ingredients.splits(name=self.ner_type, fields=tuple(fields), root=self.path)

        print('---------- NYT INGREDIENTS NER ---------')
        print('Train size: %d' % (len(train)))
        print('Validation size: %d' % (len(val)))
        print('Test size: %d' % (len(test)))

        # Build vocab
        inputs.build_vocab(train, val, max_size=50000)
        # , vectors=[GloVe(name='6B', dim='200'), CharNGram()])
        tags.build_vocab(train.label)

        logger.info('Input vocab size:%d' % (len(inputs.vocab)))
        logger.info('Tagset size: %d' % (len(tags.vocab)))

        self.inputs = inputs
        self.tags = tags

        self.dictionary.word2idx = dict(self.inputs.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.tag_vocab = self.tags.vocab.stoi

        # Get iterators
        self.train, self.valid, self.test = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.train.repeat = False
        self.valid.repeat = False
        self.test.repeat = False

        """
        return {
            'task': 'nyt_ingredients.ner',
            'iters': (self.train, self.val, self.test),
            'vocabs': (inputs.vocab, tags.vocab)
        }
        """

        """
        # Define the fields associated with the sequences.
        self.WORD = data.Field(init_token="<bos>", eos_token="<eos>")
        self.NER_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
        # Download and the load default data.
        train_data, val_data, test_data = datasets.sequence_tagging.SequenceTaggingDataset.splits(root=self.path,
                                                                fields=(('word', self.WORD), ('nertag', self.NER_TAG)))

        self.WORD.build_vocab(train_data.word, min_freq=3)
        self.dictionary.word2idx = dict(self.WORD.vocab.stoi)
        self.dictionary.idx2word = list(self.dictionary.word2idx.keys())
        self.NER_TAG.build_vocab(train_data.udtag)
        self.ner_vocab = self.NER_TAG.vocab.stoi        
        """

    def __len__(self):
        return len(self.inputs.vocab.itos)


logger = logging.getLogger(__name__)


class Ingredients(SequenceTaggingDataset):
    # New York Times ingredients dataset
    # Download original at https://github.com/NYTimes/ingredient-phrase-tagger

    urls = ['https://raw.githubusercontent.com/kolloldas/torchnlp/master/data/nyt/nyt_ingredients_ner.zip']
    dirname = ''
    name = 'nyt_ingredients_ner'

    @classmethod
    def splits(cls, fields, root=".data", train="train.txt",
               validation="valid.txt", name='nyt_ingredients_ner',
               test="test.txt", **kwargs):
        """Downloads and loads the NYT ingredients NER data in CoNLL format
        """

        Ingredients.name = name

        return super(Ingredients, cls).splits(
            fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)


if __name__ == "__main__":
    # remember this can work as a mtl problem where learn to predict the udtag and postag
    batch_size = 100
    nerc = NERCorpus(batch_size)
    # when i was returning - nerc_batch['iters'][0]
    # nerc_batch = nerc.get_data(batch_size=batch_size)

    print(nerc.tag_vocab)

    #print(nerc.dictionary.word2idx.keys())
    cnt = 0
    for i, (batch) in enumerate(iter(nerc.train)):
        cnt += batch.text.size(1)
    print(cnt)

