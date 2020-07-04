# -*- coding: utf-8 -*-
"""
 Simple example showing evaluating embedding on word_similarity datasets
"""
import logging
from six import iteritems
from web.evaluate import evaluate_on_all
import torch.utils.data
from web.similarity import fetch_MEN, fetch_WS353,\
    fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from web.analogy import fetch_google_analogy, \
    fetch_msr_analogy, fetch_semeval_2012_2, fetch_wordrep
from models.autoencoders import *
import argparse
from torch import optim
from collections import defaultdict
from helpers import *
from models.model_helpers import *
import sklearn
import random
import warnings
warnings.filterwarnings("ignore")



analogy = False
flatten = lambda l: [item for sublist in l for item in sublist]

if analogy:
    tasks = {
        "MSR": fetch_msr_analogy(),
        "GOOGLE": fetch_google_analogy(),
        "SEMEVAL": fetch_semeval_2012_2(),
        "WORDREP": fetch_wordrep()
    }
else:
    tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "SIMLEX999": fetch_SimLex999(),
        "MTURK": fetch_MTurk(),
        "RG65": fetch_RG65(),
        "RW": fetch_RW()
    }



# need to return batch pairs and first word in word pairs for all batches.
# returns a list of lists if analogy category chosen since it outputs e.g jj_adj = [3, 5]
# therefore, need to softmax output for both, OR have a sigmoid that gives one at those indices
# e.g [3, 5] == 00101

def str2int(y):
    y_uniq_name = list(set(y))
    # if using 'category' it will be like jjr_vrb
    if '_' in y[0]:
        y_pair = [ys.split("_") for ys in y]
        y_uniq_name = list(set(flatten(y_pair)))
        y_uniq_name = dict(zip(y_uniq_name, range(len(y_uniq_name))))
        y_labels = [[y_uniq_name[target[0]],y_uniq_name[target[1]]] for target in y_pair]
    # e.g noun, verb etc.
    else:
        y_uniq_name = dict(zip(y_uniq_name, range(len(y_uniq_name))))
        y_labels = [y_uniq_name[target] for target in y]
    return (y_labels)

def loop_similarity(data, name, unique_embeddings, training_data):
    for i in range(data.X.shape[0]):
        word_1 = np.vstack(unique_embeddings[data.X[i][0]])
        word_2 = np.vstack(unique_embeddings[data.X[i][1]])
        y = data.y[i]
        training_data[name]['x1'].append(word_1)
        training_data[name]['x2'].append(word_2)
        training_data[name]['y'].append(y)
    return training_data


# remember my saved analogy embeddings might not contain all the
# .X[:,2] and y strings so be careful
def loop_analogy(data, name, unique_embeddings, training_data):
    for i in range(data.X.shape[0]):
        word_1 = np.vstack(unique_embeddings[data.X[i][0]])
        word_2 = np.vstack(unique_embeddings[data.X[i][1]])
        word_3 = np.vstack(unique_embeddings[data.X[i][2]])
        word_4 = np.vstack(unique_embeddings[data.y[i]])
        y = data.category[i]
        y_aux = data.category_high_level[i]
        training_data[name]['x1'].append(word_1)
        training_data[name]['x2'].append(word_2)
        training_data[name]['x1'].append(word_3)
        training_data[name]['x2'].append(word_4)
        for i in range(2):
            training_data[name]['y'].append(y)
            training_data[name]['y_aux'].append(y_aux)
    training_data[name]['y'] = str2int(training_data[name]['y'])
    training_data[name]['y_aux'] = str2int(training_data[name]['y_aux'])
    return training_data


def mtl_tranology_data(embedding_path):
    unique_embeddings = get_data('conc', embedding_path = embedding_path)
    training_data = {k: {'x1': [], 'x2': [], 'y': [], 'y_aux': []} for k in tasks.keys()}
    datasets = {k: None for k in tasks.keys()}
    for name, data in tasks.items():
        training_data = loop_analogy(data, name, unique_embeddings, training_data)
        #print(training_data[name].keys())
        x1 = torch.from_numpy(np.array(training_data[name]['x1'])).squeeze(2)
        x2 = torch.from_numpy(np.array(training_data[name]['x2'])).squeeze(2)
        x = torch.cat([x1, x2], 1)
        y = torch.LongTensor(np.vstack(training_data[name]['y']))
        y_aux = torch.LongTensor(np.vstack(training_data[name]['y_aux']))
        dataset = torch.utils.data.TensorDataset(x, y, y_aux)
        datasets[name] = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return (tasks, datasets)

def mtl_trasim_data(embedding_path):

    unique_embeddings = get_data('conc', embedding_path = embedding_path)
    training_data = {k: {'x1': [], 'x2': [], 'y': []} for k in tasks.keys()}
    datasets = {k: None for k in tasks.keys()}
    for name, data in tasks.items():
        training_data = loop_similarity(data, name, unique_embeddings, training_data)
        x1 = torch.from_numpy(np.array(training_data[name]['x1'])).squeeze(2)
        x2 = torch.from_numpy(np.array(training_data[name]['x2'])).squeeze(2)
        x = torch.cat([x1,x2],1)
        #print(np.array(training_data[name]['y']).shape)
        y_norm = normalize(np.vstack(training_data[name]['y']))

        # REMEMBER THIS !!!
        y = torch.FloatTensor(y_norm).squeeze(1) # squeeze needed for KL


        dataset = torch.utils.data.TensorDataset(x, y)
        datasets[name] = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return (tasks, datasets)

def get_data(meta_type, embedding_path,
                 svd_type='truncated', dims = 200):

        embedding_dictionary = load_embeddings(embedding_path)
        vectors = []

        for i, (name, words) in enumerate(embedding_dictionary.items()):
            word_v = np.vstack(list(words.values()))
            vectors.append(word_v)
            if i + 1 == len(embedding_dictionary):
                word_vecs =  np.hstack(vectors)

        if meta_type == 'svd':
            if svd_type == 'standard':
                svd_vecs = np.linalg.svd(word_vecs)
            else:
                svd =sklearn.decomposition.TruncatedSVD(n_components=dims, n_iter=15, random_state=42)
                svd.fit(word_vecs)
                vecs = svd.transform(word_vecs)

        elif meta_type == 'conc':
            vecs = word_vecs
            pass

        # FINISH !
        elif meta_type == 'average':
            word_vecs = np.mean(vectors, axis= 0)
            vecs = np.pad(word_vecs, mode='max')

        all_vectors = dict(zip(list(words), vecs))

        # return word_vec dictionary and batched tensors
        return (all_vectors)

def get_vectors(args):
    if args.train:
        if args.meta != None:
            if 'multi-task' in args.meta:
                vectors = multitask_train()
            else:
                vectors = get_data(args.meta, embedding_path=args.embedding_path)
        else:
            if args.multi_task:
                vectors = multitask_train()
            else:
                vectors = train()
    else:
        if args.mt_ae_path!=None:
            vectors = load_embeddings(args.mt_ae_path)
        else:
            vectors = load_embeddings(args.ae_path)

    return (vectors)

def mtl_test(args, train_batches, model, embeddings):

    model.eval()

    # everything done after training
    embedding_pairs = []
    for dataset, vectors in train_batches.items():
        emb_x1 = [] ; emb_x2 = [];   ys = []

        for i, (x,y) in enumerate(vectors):

            x_split = torch.split(x, 1700, dim=1)  # it returns a tuple
            x1, x2 = list(x_split)  # convert to list if you want

            if args.model == 'daeme':
                x_left, x_right = list(torch.split(x1, 850, dim=1))
                x2_left, x2_right = list(torch.split(x2, 850, dim=1))
                x1_left, x1_right = Variable(x_left, requires_grad=False), Variable(x_right, requires_grad=False)
                x2_left, x2_right = Variable(x2_left, requires_grad=False), Variable(x2_right, requires_grad=False)
            else:
                x1 = Variable(x1, requires_grad=False)
                x2 = Variable(x2, requires_grad=False)

            if args.cuda:
                x1, x2 = x1.cuda(), x2.cuda()

            if args.model == 'daeme':
                yhat1_left = model.get_embedding(x1_left)
                yhat1_right = model.get_embedding(x1_right)
                yhat2_left = model.get_embedding(x2_left)
                yhat2_right = model.get_embedding(x2_right)
                yhat1 = torch.cat((yhat1_left, yhat1_right), 1)
                yhat2 = torch.cat((yhat2_left, yhat2_right), 1)
            else:
                yhat1 = model.get_embedding(x1)
                yhat2 = model.get_embedding(x2)

            yhat1 = yhat1.cpu() if args.cuda else yhat1
            yhat2 = yhat2.cpu() if args.cuda else yhat2

            # added 2nd condition because y wasn't matching yhat1,yhat2 shapes
            if yhat1.data.numpy().shape == yhat2.data.numpy().shape:
                # and y.data.numpy().shape[0] == yhat2.data.numpy().shape[0]
                emb_x1.append(yhat1.data.numpy())
                emb_x2.append(yhat2.data.numpy())
                # expand dims used because kl divergence required squeeze on line 128
                ys.append(np.expand_dims(y.numpy(), axis=1))

        ys = np.vstack(ys)
        emb_pair_x1 = np.vstack(emb_x1)
        emb_pair_x2 = np.vstack(emb_x2)

        spearman_score = get_spearman(emb_pair_x1, emb_pair_x2, ys)
        embeddings[dataset] = {'x1':emb_pair_x1, 'x2':emb_pair_x2, 'spearman':spearman_score}

    # Calculate results using helper function
    print('{0}-{1}-{2} MTAE '.format(args.loss, args.mt_loss, args.dist))
    print
    for name, data in embeddings.items():
        print("Spearman correlation of scores on {} {}".format(name, data['spearman']))

    return embeddings

def train():

    print("Retrieving Training Data....")
    word_vecs, train_batches = get_training_data(args.embedding_path)
    print("Training Data Retrieved")


    args.input_dim = list(word_vecs.values())[0].shape[0]
    args.output_dim = args.input_dim

    print("Creating or Loading MTL Model...")
    model = get_model(args)
    print("MTL Model Created")

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = get_loss(args.loss)

    if args.multi_task:
        mt_criterion = get_loss('nll')

    if args.save_loss:
        losser = []

    print("Training Started..")
    for epoch in range(args.epochs):
        for i, (x) in enumerate(train_batches):

            # since we concatenate last vector at end
            x1, x2 = Variable(x, requires_grad=True),  Variable(x, requires_grad=False)

            if args.cuda:
                x1, x2 = x1.cuda(), x2.cuda()

            # ----------------- FINISH --------------- #
            if args.multi_task:
                y = Variable(y, requires_grad=False)

            yhat = model(x1)

            if args.multi_task:
                yhat_mt = model.forward_multi(x1,x2)
                criterion_mt = mt_criterion(yhat_mt, y)
                criterion_recon = criterion(yhat, x2)
                loss = criterion_mt + criterion_recon
            else:
                loss = criterion(yhat, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.save_loss:
                losser.append(loss.data[0])

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, epoch, loss.data[0]))

    predictions = []
    for i, (x) in enumerate(train_batches):
        x = Variable(x, requires_grad=False)
        if args.cuda:
            x = x.cuda()
        yhat = model.get_embedding(x)
        yhat = yhat.cpu() if args.cuda else yhat
        predictions.append(yhat)


    preds = torch.cat(predictions).data.numpy()
    embeddings = dict(zip(list(word_vecs.keys()),preds))

    if args.save_model:
        print("Saving Embeddings")
        save_embeddings(embeddings, args.ae_model_dir + args.loss +
                        '_loss_' + 'caeme_' + args.model + '_embeddings')
    if args.save_loss:
        loss_dict = dict(zip(list(word_vecs.keys()),  losser))
        save_embeddings(loss_dict, args.ae_model_dir + args.loss +
                        '_loss_' + 'caeme_' + args.model + '_embeddings_loss')

    return (embeddings)


def multitask_train():

    retriever =  mtl_tranology_data if 'analogy' in args.embedding_path else mtl_trasim_data
    # print(args.embedding_path)
    tasks, train_batches = retriever(args.embedding_path)

    args.input_dim = 200
    args.output_dim = args.input_dim

    model = get_model(args)

    if args.cuda:
        model = model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = get_loss(args.loss)
    mt_criterion = get_loss(args.mt_loss)

    embeddings = defaultdict()

    if args.save_loss:
        losser = []

    for epoch in range(args.epochs):
        for dataset, vectors in train_batches.items():
            for i, (x,y) in enumerate(vectors):

                x_split = torch.split(x, 1700, dim=1)  # it returns a tuple
                x1, x2 = list(x_split)  # convert to list if you want

                # since we concatenate last vector at end

                if args.model == 'daeme':
                    x_left, x_right = list(torch.split(x1, 850, dim=1))
                    x2_left, x2_right = list(torch.split(x2, 850, dim=1))
                    x1_left, x1_right = Variable(x_left, requires_grad=True), Variable(x_right, requires_grad=True)
                    x2_left, x2_right = Variable(x2_left, requires_grad=True), Variable(x2_right, requires_grad=True)
                    x1_left_true, x1_right_true = Variable(x_left, requires_grad=False), Variable(x_right, requires_grad=False)
                else:
                    x1_hat = Variable(x1, requires_grad=False)

                x1, x2, y = Variable(x1, requires_grad=True), Variable(x2, requires_grad=True), Variable(y, requires_grad=False)

                if args.cuda:
                    x1, x2, x1_hat, y = x1.cuda(), x2.cuda(), x1_hat.cuda(), y.cuda()
                    if args.model == 'daeme':
                        x1_left, x1_right, x1_left_true, x1_right_true = x1_left.cuda(),\
                                                                       x1_right.cuda(),\
                                                                       x1_left_true.cuda(),\
                                                                       x1_right_true.cuda()
                        x2_left, x2_right = x2_left.cuda(), x2_right.cuda()

                if args.model == 'daeme':
                    yhat_left = model(x1_left,'left')
                    yhat_right = model(x1_right,'right')
                    criterion_recon = criterion(yhat_left, x1_left_true) + criterion(yhat_right, x1_right_true)
                    # small bit tricky because for decoupling x1 more dims than linear layer setup.
                    yhat_left_mt = model.forward_multi(x1_left, x2_left)
                    yhat_right_mt = model.forward_multi(x1_right, x2_right)
                    criterion_mt = mt_criterion(yhat_left_mt, y) + mt_criterion(yhat_right_mt, y)
                else:
                    yhat = model(x1)
                    criterion_recon = criterion(yhat, x1_hat)
                    yhat_mt = model.forward_multi(x1, x2)
                    if args.mt_loss == 'kl': yhat_mt = yhat_mt.squeeze(); y = y.squeeze()
                    criterion_mt = mt_criterion(yhat_mt, y)


                if args.loss_combo == 'random':
                    loss = random.choice([criterion_mt,criterion_recon])
                # synchronous
                else:
                    # (-criterion_mt) since dice needs to maximized
                    if args.mt_loss == 'dice':
                        loss = criterion_recon - criterion_mt
                    elif args.mt_loss == 'nll':
                        loss = criterion_recon + criterion_mt
                    else:
                        # should this be minus with no abs ?
                        loss =  criterion_recon - criterion_mt

                optimizer.zero_grad()

                if args.loss_combo == 'async':
                    criterion_mt.backward()
                    criterion_recon.backward()
                else:
                    loss.backward()

                optimizer.step()

                if args.save_loss:
                    losser.append(loss.data[0])

            if epoch % args.log_interval == 0:
                # ===================log========================
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epoch, loss.data[0]))


    embeddings = mtl_test(args, train_batches, model, embeddings)

    """
    embeddings = dict(zip(list(word_vecs.keys()), preds))

    if args.save_model:
        save_embeddings(embeddings, args.ae_model_dir + args.loss + '_multi_task' +
                        '_loss_' + 'caeme_' + args.model + '_embeddings')
    if args.save_loss:
        loss_dict = dict(zip(list(word_vecs.keys()), losser))
        save_embeddings(loss_dict, args.ae_model_dir + args.loss +
                        '_loss_' + 'caeme_' + args.model + '_embeddings_loss')
    """
    return (embeddings)


def get_spearman_scores(args):

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
    vectors = get_vectors(args)

    if args.meta == 'multi-task':
        return 0

    # bit redundant reloading datsets if already used for multi-task learning ...
    if args.evaluate_all:
        evaluate_on_all(vectors)
    else:
        # Print sample data
        for name, data in iteritems(tasks):
            print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))
        # Calculate results using helper function
        for name, data in iteritems(tasks):
            print("Spearman correlation of scores on {} {}".format(name, evaluate_similarity(vectors, data.X, data.y)))


if __name__ == '__main__':

    task = 'word_similarity' # analogy
    emb_type = 'caeme'
    ROOT = "embeddings/" + task + '/'
    EMB_PATH = ROOT + task + "_word_embeddings"
    AE_PATH = ROOT + emb_type +'_embeddings'
    parse = argparse.ArgumentParser()

    # can be conc, average or svd, should be None when using AE
    # also, use multi-task when training using multi-task learning

    parse.add_argument("--meta", default='multi-task', type= str)

    parse.add_argument("--save_model",default=True, type = bool)
    parse.add_argument("--save_loss",default=True, type = bool)

    # evaluate all, meaning evaluate anaology, word_similarity etc,
    #  false just means choose only word_similarity datasets

    parse.add_argument("--evaluate_all", default=True, type = bool)


    # evaluate_downstream overrides evaluate all NOT USED
    parse.add_argument("--evaluate_downstream", default=False, type = bool)
    parse.add_argument("--nmt", default=False, type = bool)
    parse.add_argument("--pos", default=False, type = bool)
    parse.add_argument("--dep", default=False, type = bool)

    parse.add_argument("--embedding_path", default=EMB_PATH,
                       help="default: embeddings/word_similarity/similarity_word_embeddings")
    parse.add_argument("--ae_model_dir",default=ROOT, type = str)
    parse.add_argument("--model_dir",default=ROOT, type = str)
    parse.add_argument("--train",default=True, type = bool)
    parse.add_argument("--ae_path",default=AE_PATH,
                       help = 'default: embeddings/word_similarity/caeme_embeddings')

    # linear, euclidean, cosine
    parse.add_argument("--dist", default='euclidean', type=str)
    # caeme or daeme
    parse.add_argument("--model", default=emb_type, type=str)
    # mse, cosine, kl, manhattan
    parse.add_argument("--loss", default='cosine', type=str)
    # dice, nll or mse
    parse.add_argument("--mt_loss", default='mse', type=str)
    # random, async, sync
    parse.add_argument("--loss_combo", default='async', type=str)

    parse.add_argument("--seed", default=1, type=int)
    parse.add_argument("--epochs", default= 10, type=int)
    parse.add_argument("--optimizer", default='adam', type=str)
    parse.add_argument("--input_dim", default = None, type=int)
    parse.add_argument("--output_dim", default = None, type=int)
    parse.add_argument("--hidden_dim", default=300, type=int)
    parse.add_argument("--activations", default='tanh', type=str)
    parse.add_argument("--cuda", default=False, type=bool)
    parse.add_argument("--num_layers", default=1, type=int)
    parse.add_argument("--log_interval", default=2, type=int)

    parse.add_argument("--multi-task", default=True, type=bool)

    args = parse.parse_args()
    torch.manual_seed(args.seed)

    models = ['caeme', 'daeme']
    distances = ['cosine','euclidean', 'linear']
    losses = ['mse', 'cosine', 'kl', 'manhattan']
    mt_losses = ['mse', 'kl', 'dice']

    for model in models:
        args.model = model
        for distance in distances:
            args.dist = distance
            for mt_loss in mt_losses:
                args.mt_loss = mt_loss
                for loss in losses:
                    args.loss = loss
                    get_spearman_scores(args)

    #get_spearman_scores(args)


