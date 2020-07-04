import argparse
from macros import *

MODEL = 'LSTM'

# WIKITEXT3 IS NOT CLEAN SO VOCABULARY IS MUCH LARGER, USE WIKITEXT2 FOR TESTING

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 and Wikitext-3 RNN/LSTM Language Model')

# training, data and preprocess params
parser.add_argument('--data', type=str, default=POS_ROOT,
                    help='location of the data corpus')
parser.add_argument('--vocab_limit', type=int, default=int(5e4),
                    help='vocabulary limit (important for en8 and wiki103)')
parser.add_argument('--lcs', type=bool, default=False,
                    help='if true, it tries to assign idx for oov words which have the closes lcs'
                         'to words within the vocabulary limit based on term frequency')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
# network params
parser.add_argument('--model', type=str, default=MODEL,
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
# if pretrain true make sure to keep emsize and nhid = 300
parser.add_argument('--pretrained', default=None,
                    help='size of word embeddings: None, all, glove, skipgram, lexvec, numberbatch, fasttext')
parser.add_argument('--tunable', type=bool, default=False)
parser.add_argument('--target_pretrained', default=None,
                    help='in the case of machine translation where there is two vocabularies'
                         'size of word embeddings: None, word or subword')
parser.add_argument('--tune', default= False,
                    help='tunes pretrained embeddings if set true')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_method', type=str, default='standard',
                    help='standard, gaussian, variational, concrete, curriculum')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--dec_nhid', type=int, default=200,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--dec_output_size', type=int, default=200,
                    help='decoder output size')
parser.add_argument('--optimizer', type=str, default='amsgrad',
                    help='when set to None, it picks vanilla sgd with decayed learning rate')
parser.add_argument('--dec_optimizer', type=str, default='amsgrad',
                    help='when set to None, it picks vanilla sgd with decayed learning rate')
parser.add_argument('--scheduler', type=str, default='cosine_anneal',
                    help='None, cosine_anneal, lro, multi_step')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# utils params
parser.add_argument('--cuda', default=True, action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_model', type=bool, default=True,
                    help='whether to save the model or not')
parser.add_argument('--save_test_batch', type=bool, default=False,
                    help='save numpy array of target indices with vocab')
parser.add_argument('--save', type=str, default='./save_models/'+MODEL.lower()+'.pt',
                    help='path to save the final model')
parser.add_argument('--save_losses', type=bool, default=False,
                    help='stores probabilities for each token in a sequence'
                         'so we can compare variance as a function of time')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# print params
parser.add_argument('--check_grad', default=False, type=bool,
                    help='prints out gradients of network to make sure gradients are being updated for all parameters')


args = parser.parse_args()