""" Simple example showing evaluating embedding on word_similarity datasets """

import torch.utils.data
import argparse
from torch import optim
from models.autoencoders import *
import logging
from web.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999,\
    fetch_MTurk, fetch_RG65, fetch_RW
from web.evaluate import get_vector_pairs
import pickle
import numpy as np


BATCH_SIZE = 32


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)

    print(tensor.size())

    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def batch_vecs(vecs, x, y):
    embedding_dictionary = load_embeddings('similarity_word_embeddings')
    x1,x2 = get_vector_pairs(vecs, x, y)
    x1, x2 = torch.FloatTensor(x1), torch.FloatTensor(x2)
    x = torch.utils.data.DataLoader((x1,x2), batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    return(x)

def load_vectors(x1, x2, source_embedding='glove',target_embedding='sg', tasks=None):
    #task_vectors = {"MEN":{},"WS353": {},"SIMLEX999": {}}
    #x1,x2 = load_vector(vecs, x, y)
    x1, x2 = torch.FloatTensor(x1), torch.FloatTensor(x2)
    x = torch.utils.data.DataLoader((x1,x2), batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    return(x)

def save_embeddings(embeddings, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)

def load_embeddings(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_dataset_vectors(vectors, dset_pairs):
    pair1 = []; pair2 = []
    dset_pairs = dset_pairs['X']
    X_1, X_2 = dset_pairs[:,0], dset_pairs[:, 1]
    for (x1, x2) in zip(X_1, X_2):
        pair1.append(vectors[x1])
        pair2.append(vectors[x2])
    pair1 = np.array(pair1)
    pair2 = np.array(pair2)
    return (torch.utils.data.DataLoader((pair1,pair2), batch_size=BATCH_SIZE, shuffle=True, num_workers=6))


def get_relational_vectors(embedding_dictionary, target):

    source_vecs = []
    words = list(embedding_dictionary.values())[0]
    #print(len(words))

    for (name, vecs) in embedding_dictionary.items():
        print(name.lower())
        if target.lower() not in name.lower():
            l = list(vecs.values())
            source_vecs.append(np.vstack(l))
        else:
            target_vec = np.vstack(list(vecs.values()))

    source_vecs = np.hstack(source_vecs)

    print("SOURCE")
    print(source_vecs.shape)
    print("TARGET")
    print(target_vec.shape)

    # remember to cut end off for target
    dims = (source_vecs.shape[1],target_vec.shape[1])
    vectors = np.hstack([source_vecs, target_vec])

    # shuffle needs to be false otherwise word order will be jumbled up so word keys will be wrong
    vec_batches = torch.utils.data.DataLoader(vectors, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return (vec_batches, words, dims)


# should just retrieve the saved relational autoencoder embeddings to then test on the datasets.
# NOT trying to use all embeddings to train relational autoencoder while looping through datasets.
def get_vectors(tasks=None):

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
    # Fetch all embeddings (warning: it might take few minutes)
    all_vectors = load_embeddings('embeddings/similarity_word_embeddings')

    dsets = {
        "MEN": fetch_MEN(),"WS353": fetch_WS353(),"SIMLEX999": fetch_SimLex999(),
        "MTURK": fetch_MTurk(),"RG_65": fetch_RG65(), "RW": fetch_RW()
    }
    dset_vectors = {
        "MEN": None, "WS353": None, "SIMLEX999": None,
        "MTURK": None, "RG_65": None, "RW": None
    }

    for name, dset_vec in dset_vectors.items():
        dset_vectors[name] = dict(zip(all_vectors.keys(),[None]*len(all_vectors.keys())))

    # Calculate results using helper function
    for vec_name, vec in all_vectors.items():
        for (dset_name, words) in dsets.items():

            dset_vectors[dset_name][vec_name] = get_dataset_vectors(all_vectors[vec_name], dsets[dset_name])

    """
    for name, data in iteritems(tasks):
        #vecs = {source_embedding:w_glove,target_embedding:w_skipgram}
        x_source, x_target = batch_vecs(all_vectors, data.X, data.y)
        task_vectors[name] = (x_source, x_target)
    """

    return (dset_vectors)


def train_relational_embeddings(args):

    task = args.dataset.upper()
    print(args.vector_pairs.keys())

    embedding_dictionary = load_embeddings(args.model_dir+ 'similarity_word_embeddings')

    if args.save_model:
        embeddings = dict(zip(args.target_vectors,[None]*len(args.target_vectors)))

    if args.save_loss:
        loss_dict = dict(zip(args.target_vectors,[None]*len(args.target_vectors)))

    for target_vec in args.target_vectors:
        try:
            args.vector_pairs, words, dims = get_relational_vectors(embedding_dictionary, target_vec)
            args.input_dim, args.output_dim = dims
        except:
            ValueError(
                "Seems like something went wrong...make sure to pick an "
                       "existing target vector to pick from"
                       )

        model = Autoencoder(args)

        if args.cuda:
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        if args.save_loss:
            losser = []

        for epoch in range(args.epochs):
            for i, (x) in enumerate(args.vector_pairs):

                # since we concatenate last vector at end
                y, x = Variable(x[:, :dims[1]],requires_grad=False), \
                       Variable(x[:, :-dims[1]], requires_grad=True
                                )
                if args.cuda: x, y = x.cuda(), y.cuda()

                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.save_loss:
                    losser.append(loss.data[0])

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, epoch, loss.data[0]))

        predictions = []
        for i, (x) in enumerate(args.vector_pairs):
            y, x = Variable(x[:, :dims[1]], requires_grad=False), \
                   Variable(x[:, :-dims[1]], requires_grad=True)
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            yhat = model.get_embedding(x)
            yhat = yhat.cpu() if args.cuda else yhat
            predictions.append(yhat)

        if args.save_model:
            preds = torch.cat(predictions)
            assert len(words) == len(preds)
            embeddings[target_vec] = preds
            embeddings['words'] = words
        if args.save_model:  loss_dict[target_vec] = losser

    if args.save_model:
        save_embeddings(
            embeddings, args.relational_model_dir+args.loss+'_loss'+'relational_'+args.model+'_embeddings'
                        )
    if args.save_loss:
        save_embeddings(
            loss_dict, args.model_dir+'relational_'+args.model+'_embeddings_loss'
        )


def train(args):

    task = args.dataset.upper()
    print(args.vector_pairs.keys())

    embedding_dictionary = load_embeddings('similarity_word_embeddings')
    try:
        args.vector_pairs = args.vector_pairs[task]
    except:
        ValueError("You must pick the right dataset for vector pair")

    model = Autoencoder(args)

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(),lr = 0.0005)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        for i, (x1, x2) in enumerate(args.vector_pairs):
            x1, x2 = Variable(x1), Variable(x2)
            if args.cuda:
                x1, x2 = x1.cuda(), x2.cuda()
            y = model(x1)
            loss = criterion(y, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, epoch, loss.data[0]))


if __name__ == '__main__':

    ds = ['simlex', 'rareword', 'sim353', 'men', 'simverb', 'sn', 'bless']
    embeddings = ['cbow', 'glove', 'word2vec', 'hlbl', 'pmi']
    save = False

    # nhid set to 100 here to be equal to emsize given they should be equal for tied weights
    # bptt decides the number of time steps (look at get_batch in gru_helper)

    parse = argparse.ArgumentParser()

    # fixed dim is the set size of the hidden dimensions of the autoencoders
    # so to average over the same dimensions
    vec_pair = get_vectors()

    target_vecs = ["glove","fasttext","skipgram","hdc","lexvec","HPCA"]
    # or target_vecs = ["glove"]

    parse.add_argument("--vector_pairs",default=vec_pair)
    parse.add_argument("--target_vectors", default=target_vecs)
    parse.add_argument("--encode_relational_vectors", default=True, type=bool)
    parse.add_argument("--dataset",default='men')

    # options are mse or cosine
    parse.add_argument("--loss", default='cosine')

    parse.add_argument("--save_model",default=False, type = bool)
    parse.add_argument("--save_loss",default=False, type = bool)
    parse.add_argument("--relational_model_dir", default='relational_embeddings/', type = str)
    parse.add_argument("--model_dir",default='embeddings/', type = str)

    parse.add_argument("--model", default='ae', type=str)

    parse.add_argument("--pretrain", default=True, type=int)
    parse.add_argument("--data", default=ds)
    parse.add_argument("--attention", default=False, type=bool)
    parse.add_argument("--attention_width", default=5, type=int)
    parse.add_argument("--embeddings", default=embeddings)
    parse.add_argument("--clip", default=0.0001, type=float)
    parse.add_argument("--lr", default=0.001, type=float)
    parse.add_argument("--num_layers", default=True, type=int)
    parse.add_argument("--cuda", default=False, type=bool)
    parse.add_argument("--epochs", default=50, type=int)
    parse.add_argument("--optimizer", default='adam', type=str)
    parse.add_argument("--log-interval", default=10, type=int)
    parse.add_argument("--seed", default=1, type=int)

    parse.add_argument("--input_dim", default = None, type=int)
    parse.add_argument("--output_dim", default = None, type=int)
    parse.add_argument("--hidden_dim", default=300, type=int)
    parse.add_argument("--activations", default='tanh', type=str)

    args = parse.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.encode_relational_vectors:
        train_relational_embeddings(args)
    else:
        train(args)