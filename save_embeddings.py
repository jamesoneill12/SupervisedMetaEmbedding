""" Saving all vectors for all unique words for all datasets """
import logging
from six import iteritems
from web.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from web.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2, fetch_wordrep
from web.downstream import Downstream
from web.embeddings import fetch_GloVe, fetch_FastText, fetch_SG_GoogleNews, \
    fetch_HDC, fetch_LexVec, fetch_HPCA
from web.evaluate import evaluate_similarity
import numpy as np
from loaders.embedding import Embedding
import pickle
import argparse


def save_embeddings(embeddings, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)


# assume w is a list with w[0] = source, w[1] = target
def get_vector_pairs(words, embedding):
    if isinstance(embedding, dict):
        embedding = Embedding.from_dict(embedding)
    # mean vector is the defult if word not found
    # print(words)
    mean_vector = np.mean(embedding.vectors, axis=0, keepdims=True)
    x = {word: embedding.get(word, mean_vector) for word in words}
    return x


def similarity_words(tasks):
    all_words = []
    for dataset, w in tasks.items():
        # print(type(w.X))
        if type(w.X) == dict:
            words = []
            for k, v in w.X.items():
                words += list(set(list(v[:, 0]) + list(v[:, 1])))
            words = list(set(words))
        else:
            words = list(set(list(w.X[:,0])+list(w.X[:, 1])))
        all_words.extend(words)
    return all_words


def analogy_words(tasks):
    all_words = []
    for dataset, w in tasks.items():
        if type(w.X) == dict:
            words = []
            for k, v in w.X.items():
                words += list(set(list(v[:, 0]) + list(v[:, 1]) + list(v[:, 2])))
            for k, v in w.y.items():
                words += list(set(list(v)))
            words = list(set(words))
        else:
            words = list(set(list(w.X[:,0])+list(w.X[:, 1]) + list(v[:, 2])))
            for k, v in w.y.items():
                words += list(set(list(v)))
        all_words.extend(words)
    return all_words


def downstream_words(tasks):
    all_words = []
    for dataset, w in tasks.items():
        all_words+=w
    return all_words


def get_unique_words(tasks):
    # assuming tasks is a DICT here
    if "MEN" in tasks:
        all_words = similarity_words(tasks)
    elif "MSR" in tasks:
        all_words = analogy_words(tasks)
    else:
        all_words = downstream_words(tasks)
    unique_words = list(set(all_words))
    return unique_words


def get_wordvec_dictionaries(tasks, vector_mods, save=None):

    all_words = similarity_words(tasks) if "MEN" in tasks else downstream_words(tasks)
    unique_words = list(set(all_words))
    word_embeddings =  dict(zip(list(vector_mods.keys()), [None]*len(vector_mods.keys())))

    for (model_name, embedding) in vector_mods.items():
        word_embeddings[model_name] = get_vector_pairs(unique_words, embedding)

    if save!=None:
        save_embeddings(word_embeddings, save+'_word_embeddings')


def get_wordvec_dictionary(words, model_name, word_embeddings, embedding):
    word_embeddings[model_name] = get_vector_pairs(words, embedding)
    return word_embeddings


def get_downstream_dictionary(tasks, embeddings, save = None):

    word_embeddings = dict(zip(list(embeddings.keys()), [None] * len(embeddings.keys())))
    for name, vocab in tasks.items():
        for (model_name, embedding) in embeddings.items():
            word_embeddings[model_name] = get_vector_pairs(vocab, embedding)
        if save!=None:
            save_embeddings(
                word_embeddings, 'embeddings/'+name.lower()+'/'+ name.lower()+'embeddings'
            )


def get_tasks(ttype='word_similarity'):
    if ttype.lower() == 'word_similarity':
        # Define tasks
        tasks = {
            "MEN": fetch_MEN(),
            "WS353": fetch_WS353(),
            "SIMLEX999": fetch_SimLex999(),
            "MTURK": fetch_MTurk(),
            "RG_65": fetch_RG65(),
            "RW": fetch_RW()
        }
    elif ttype.lower() == 'analogy':
        tasks = {
            "MSR": fetch_msr_analogy(),
            "GOOGLE": fetch_google_analogy(),
            "SEMEVAL": fetch_semeval_2012_2(),
            "WORDREP": fetch_wordrep()
        }
    else:
        ds = Downstream()
        if ttype.lower() == 'pos':
            tasks = {"POS": ds.fetch_POS_vocab()}
        elif ttype.lower() == 'trec':
            tasks = {"TREC": ds.fetch_TREC_vocab()}
        elif ttype.lower() == "sent":
            tasks = {"SENT": ds.fetch_SENT_vocab()}
        elif ttype.lower() == "downstream":
            tasks = {
                "POS": ds.fetch_POS_vocab(),
                "TREC": ds.fetch_TREC_vocab(),
                "SENT": ds.fetch_SENT_vocab()
            }

    return tasks


def choose_pretrain_embeddings(model_name, norm = False):
    if model_name == "GLOVE":
        embeddings = fetch_GloVe(corpus="wiki-6B", dim=300)
    elif model_name == "FASTTEXT":
        embeddings = fetch_FastText(normalize=norm)
    elif model_name == "SKIPGRAM":
        embeddings = fetch_SG_GoogleNews(normalize=norm)
    elif model_name == "LEXVEC":
        embeddings = fetch_LexVec(normalize=norm)
    elif model_name == "HPCA":
        embeddings = fetch_HPCA(normalize=norm)
    elif model_name == "HDC":
        embeddings = fetch_HDC(normalize=norm)
    return embeddings


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--save_vecs",default=True, type = bool)
    parse.add_argument("--incremental",default=True, type = bool)
    parse.add_argument("--ttype", default='analogy', type = str,
                       help = "downstream (pick all downstream), pos, trec, sent, word_similarity (all word_similarity)")

    args = parse.parse_args()

    tasks = get_tasks(args.ttype)

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    if args.incremental:

        embeddings = ["SKIPGRAM", "GLOVE", "FASTTEXT", "LEXVEC",  "HPCA", "HDC"]
        word_embeddings =  dict(zip(list(embeddings), [None]*len(embeddings)))
        unique_words = get_unique_words(tasks)

        for i, model_name in enumerate(word_embeddings.keys()):
            embeddings = choose_pretrain_embeddings(model_name)
            word_embeddings = get_wordvec_dictionary(unique_words, model_name, word_embeddings, embeddings)
            del embeddings

        if args.save_vecs != None:
            save_embeddings(word_embeddings, 'analogy_word_embeddings')

    else:
        # Fetch GloVe embedding (warning: it might take few minutes)
        embeddings = {
            "GLOVE": fetch_GloVe(corpus="wiki-6B", dim=300),
            "FASTTEXT": fetch_FastText(normalize=True),
            "SKIPGRAM": fetch_SG_GoogleNews(normalize=True),
            "LEXVEC": fetch_LexVec(),
        }
        if "MEN" in tasks or "MSR" in tasks:
            get_wordvec_dictionary(tasks, embeddings, save=args.ttype)
        else:
            get_downstream_dictionary(tasks, embeddings, save=args.ttype)




