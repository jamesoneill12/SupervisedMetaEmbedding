""" Simple example showing evaluating embedding on word_similarity datasets """
import logging
from web.evaluate import evaluate_on_all_relational
import pickle
import argparse


def load_embeddings(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_spearman_scores(args):

    # Configure logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.DEBUG, datefmt='%I:%M:%S'
                        )
    vectors = load_embeddings(args.relational_path)

    if args.concat_target:
        args.concat_target = load_embeddings('embeddings/word_similarity/word_embeddings')

    results = dict(zip(list(vectors.keys()), [None]*len(vectors.keys())))
    del results['words']

    # Calculate results using helper function
    for name, data in vectors.items():
        print(name.title())
        print
        if name !='words':
            results[name] = evaluate_on_all_relational(vectors,
                                                       target_vector=name,
                                                       concat_target=args.concat_target)


if __name__ == '__main__':

    parse = argparse.ArgumentParser()

    #target_vecs = ["glove","fasttext","skipgram","hdc","lexvec","HPCA"]
    # or target_vecs = ["glove"]

    parse.add_argument("--relational_path",default='embeddings/relational/ae_embeddings')
    parse.add_argument("--target_vectors", default='all')
    parse.add_argument("--concat_target", default=True, type= bool)
    parse.add_argument("--save_model",default=False, type = bool)
    parse.add_argument("--model", default='ae', type=str)
    parse.add_argument("--seed", default=1, type=int)

    args = parse.parse_args()

    get_spearman_scores(args)