from extrinsic_settings import *
from loaders.pos import PoSCorpus
from loaders.ner import NERCorpus
from loaders.chunking import CoNLLCorpus
from loaders.sentiment import SentimentCorpus
from macros import POS_ROOT
from trainers.train import run
import os


def get_corpus(path):
    if 'pos' in path:
        return PoSCorpus(batch_size=args.batch_size, devicer=args.cuda)
    if 'ner' in path:
        return NERCorpus(batch_size=args.batch_size, devicer=args.cuda)
    if 'chunk' in path:
        return CoNLLCorpus(batch_size=args.batch_size, devicer=args.cuda)
    if 'sent' in path:
        return SentimentCorpus(batch_size=args.batch_size)


def get_task_name(path):
    if 'pos' in path:
        return "udpos"
    if 'ner' in path:
        return "ner"
    if 'chunk' in path:
        return "chunking"
    if 'sent' in path:
        return "sentiment"


if __name__ == "__main__":
    
    paths = [POS_ROOT, NER_ROOT, CHUNKING_ROOT, SENT_ROOT]
    models = ['LSTM']
    pretrains = [None]
    optimizers = [None]  # amsgrad, sgd with lr annealing
    schedulers = [None]

    save_root = "save/"
    embeddings = ["all", None, "glove", "skipgram",
                  "lexvec", "numberbatch", "fasttext"]

    args.batch_size = 56
    task_epochs = [50, 2, 10, 10]
    intervals = [10, 50, 20, 20]
    args.control = False
    args.tunable = True

    for i, path in enumerate(paths):
        args.data = path
        args.pos = True
        args.nptb_token = False

        corpus = get_corpus(path)

        for model in models:
            for embedding in embeddings:
                args.model = model
                args.pretrained = embedding
                # if args.optimizer is None:
                args.optimizer = 'adam'
                args.epochs = task_epochs[i]
                args.log_interval = intervals[i]
                tname = get_task_name(path)
                args.results_path = "{}/{}/tunable{}_embedding{}_model{}_epochs{}.pkl".format(save_root,
                                                                                     tname, args.tunable, embedding,
                                                                                              model, args.epochs)
                if os.path.exists(args.results_path) == False:
                    run(args, corpus)