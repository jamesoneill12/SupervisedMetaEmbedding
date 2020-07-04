import torch
from web.downstream import Downstream
import argparse
import numpy as np


def get_mtl_pos_data(embedding_path= 'embeddings/pos_word_embeddings'):
    t = Downstream()
    task = t.fetch_POS()
    unique_embeddings = t.get_data('conc')
    training_data =  {k: {'x1':[], 'x2':[],'y':[]} for k in tasks.keys()}
    datasets = {k:None for k in tasks.keys()}
    for name, data in tasks:
        for i in range(data.X.shape[0]):
            word_1 = np.vstack(unique_embeddings[data.X[i][0]])
            word_2 = np.vstack(unique_embeddings[data.X[i][1]])
            y = data.y[i]
            training_data[name]['x1'].append(word_1)
            training_data[name]['x2'].append(word_2)
            training_data[name]['y'].append(y)

        x1 = torch.from_numpy(np.array(training_data[name]['x1'])).squeeze(2)
        x2 = torch.from_numpy(np.array(training_data[name]['x2'])).squeeze(2)
        x = torch.cat([x1,x2],1)
        y_norm = normalize(np.vstack(training_data[name]['y']))
        y = torch.FloatTensor(y_norm)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    return datasets


if __name__ == '__main__':

    parse = argparse.ArgumentParser()

    # can be conc, average or svd, should be None when using AE
    # also, use multi-task when training using multi-task learning

    parse.add_argument("--meta", default='multi-task', type= str)

    parse.add_argument("--save_model",default=True, type = bool)
    parse.add_argument("--save_loss",default=True, type = bool)

    # evaluate all, meaning evaluate anaology, word_similarity etc,
    #  false just means choose only word_similarity datasets

    parse.add_argument("--evaluate_all", default=False, type = bool)

    # evaluate_downstream overrides evaluate all
    parse.add_argument("--evaluate_downstream", default=False, type = bool)
    parse.add_argument("--nmt", default=False, type = bool)
    parse.add_argument("--pos", default=False, type = bool)
    parse.add_argument("--dep", default=False, type = bool)

    parse.add_argument("--ae_model_dir",default='embeddings/', type = str)
    parse.add_argument("--model_dir",default='embeddings/', type = str)
    parse.add_argument("--train",default=True, type = bool)
    parse.add_argument("--ae_path",default='embeddings/caeme_embeddings')

    # linear, euclidean, cosine
    parse.add_argument("--dist", default='euclidean', type=str)
    # caeme or daeme
    parse.add_argument("--model", default='caeme', type=str)
    # mse, cosine, kl, manhattan
    parse.add_argument("--loss", default='cosine', type=str)
    # dice, nll or mse
    parse.add_argument("--mt_loss", default='mse', type=str)
    # random, async, sync
    parse.add_argument("--loss_combo", default='async', type=str)

    parse.add_argument("--seed", default=1, type=int)
    parse.add_argument("--epochs", default= 20, type=int)
    parse.add_argument("--optimizer", default='adam', type=str)
    parse.add_argument("--input_dim", default = None, type=int)
    parse.add_argument("--output_dim", default = None, type=int)
    parse.add_argument("--hidden_dim", default=300, type=int)
    parse.add_argument("--activations", default='tanh', type=str)
    parse.add_argument("--cuda", default=False, type=bool)
    parse.add_argument("--num_layers", default=1, type=int)

    parse.add_argument("--multi-task", default=True, type=bool)

    args = parse.parse_args()
    torch.manual_seed(args.seed)

    distances = ['cosine','euclidean', 'linear']
    losses = ['mse', 'cosine', 'kl', 'manhattan']
    mt_losses = ['kl', 'mse', 'dice']

    for distance in distances:
        for mt_loss in mt_losses:
            for loss in losses:
                args.dist = distance
                args.mt_loss = mt_loss
                args.loss = loss
                get_spearman_scores(args)

    #get_spearman_scores(args)
