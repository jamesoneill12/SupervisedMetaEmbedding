import torch.nn.functional as F
from torch import nn
from models.regularizers import dropout
from models.attention import Attention
from torch.autograd import Variable
import torch
import math


# https://arxiv.org/pdf/1505.00387.pdf
class Highway(nn.Module):
    def __init__(self, sizes, num_layers, f=F.relu):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class HighwayText(nn.Module):
    """
    Embeddings need to be flattened to 2d to be used with highway network.
    """
    def __init__(self, vocab_size, hidden_size, sent_len, nlayers=2, dropout_rate=0.2,
                 class_num=None, kernel_num=None, kernel_sizes=None):
        super(HighwayText, self).__init__()

        V = vocab_size
        D = hidden_size
        C = hidden_size if class_num is None else class_num
        Ci = 1
        Co = 300 if kernel_num is None else kernel_num
        Ks = (2, 3, 4) if kernel_sizes is None else kernel_sizes

        self.sizes = [hidden_size * sent_len, hidden_size * 2,  hidden_size] \
            if nlayers == 3 else [hidden_size * sent_len, hidden_size * 2, hidden_size,  hidden_size]
        self.embed = nn.Embedding(V, D)
        self.highway = Highway(sizes=self.sizes, num_layers=nlayers, f=torch.tanh)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        # print(x.view(x.size(0), x.size(1)*x.size(2)).size())
        x = self.highway(x.view(x.size(0), x.size(1)*x.size(2)))
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

"""
Recurrent Highway Networks
"""


def highwayGate(Ws, s, gateDrop, trainable):
    h = int(Ws.size()[1] / 2)
    hh, tt = torch.split(Ws, h, 1)
    hh, tt = torch.tanh(hh), torch.sigmoid(tt)
    cc = 1 - tt
    tt = F.dropout(tt, p=gateDrop, training=trainable)
    return hh * tt + s * cc


class RHNCell(nn.Module):
    def __init__(self, embedDim, h, depth, gateDrop):
        super(RHNCell, self).__init__()
        self.h, self.depth, self.gateDrop = h, depth, gateDrop
        self.inp = nn.Linear(embedDim, 2 * h)
        self.cell = nn.ModuleList([
            nn.Linear(h, 2 * h) for i in range(depth)])

    def forward(self, x, s, trainable):
        sOut = []
        for l in range(self.depth):
            Ws = self.cell[l](s) if s is not 0 else 0
            if l == 0:
                Ws += self.inp(x)
            s = highwayGate(Ws, s, self.gateDrop, trainable)
            sOut += [s]
        return s, s, sOut


class RHNContextCell(nn.Module):
    """
    Highway Network Cell that also takes context attention vector weights as input as well
    """
    def __init__(self,embedDim, h, depth, gateDrop):
        super(RHNContextCell, self).__init__()
        self.embed_dim = embedDim
        self.h, self.depth, self.gateDrop = h, depth, gateDrop
        self.inp = nn.Linear(embedDim, 2 * h)
        self.cell = nn.ModuleList([
            nn.Linear(h, 2 * h) for i in range(depth)])

    def forward(self, x, s, ctx, trainable):
        # ctx: 20 x 800
        sOut = []
        j = 0
        for l in range(self.depth):
            # applies attention vector to input
            Ws = self.cell[l](s * ctx[:, j:self.embed_dim+j]) if s is not 0 else 0
            if l == 0:
                Ws += self.inp(x)
            s = highwayGate(Ws, s, self.gateDrop, trainable)
            sOut += [s]
            j+self.embed_dim
        return s, s, sOut


class HyperCell(nn.Module):
    def __init__(self, embedDim, h, depth, gateDrop):
        super(HyperCell, self).__init__()
        self.h, self.depth, self.gateDrop = h, depth, gateDrop
        self.inp = HyperLinear(embedDim, 2 * h)
        self.cell = nn.ModuleList([
            HyperLinear(h, 2 * h) for i in range(depth)])

    def forward(self, x, s, z, trainable):
        sOut = []
        for l in range(self.depth):
            Ws = self.cell[l](s, z[l]) if s is not 0 else 0
            if l == 0:
                Ws += self.inp(x, z[l])
            s = highwayGate(Ws, s, self.gateDrop, trainable)
            sOut += [s]
        return s, sOut


class HyperRHNCell(nn.Module):
    def __init__(self, embedDim, h, depth, gateDrop):
        super(HyperRHNCell, self).__init__()
        hHyper, hNetwork = h
        self.HyperCell = RHNCell(embedDim, hHyper, depth, gateDrop)
        self.RHNCell = HyperCell(embedDim, hNetwork, depth, gateDrop)
        self.upscaleProj = nn.ModuleList([nn.Linear(hHyper, hNetwork)
                                          for i in range(depth)])

    def initializeIfNone(self, s):
        if s is not 0: return s
        return (0, 0)

    def forward(self, x, s, trainable):
        sHyper, sNetwork = self.initializeIfNone(s)
        _, _, sHyper = self.HyperCell(x, sHyper, trainable)
        z = [self.upscaleProj[i](e) for i, e in enumerate(sHyper)]
        out, sNetwork = self.RHNCell(x, sNetwork, z, trainable)
        return out, (sHyper[-1], sNetwork[-1]), (sHyper, sNetwork)


class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(HyperLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, z):
        weight = self.weight
        z = torch.cat((z,z), 1)
        Wx = self._backend.Linear()(input, weight)*z
        Wx += self.bias.expand_as(Wx)
        return Wx

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class RecurrentHighwayText(nn.Module):

    """
    Embeddings need to be flattened to 2d to be used with highway network.
    Note: When encoder set to true then both (2) last hidden states
    pass to the decoder
    """

    def __init__(self, vocab_size, hidden_size, nlayers=2,
                 dropout_rate=0.2, class_num=None, hyper=False, lm = False,
                 attention=True, embedding=True, drop_method='standard'):
        super(RecurrentHighwayText, self).__init__()

        V = vocab_size
        D = hidden_size

        # remember to pass hidden states to decoder for hway2hway nets
        C = hidden_size if class_num is None else class_num

        self.embedding = embedding
        if self.embedding:
            self.embed = nn.Embedding(V, D)
        self.dropout = dropout(p=dropout_rate, dim=D, method=drop_method)
        # self.dropout = nn.Dropout(dropout_rate)
        self.nlayers = nlayers
        self.att = attention
        self.lm = lm

        if attention is True:
            self.fc1 = nn.Linear(C, C)

        if hyper:
            self.rnnhighway = HyperRHNCell(embedDim=hidden_size, h=hidden_size,
                                  depth=nlayers, gateDrop=dropout_rate)
        else:
            self.rnnhighway = RHNCell(embedDim=hidden_size, h=hidden_size,
                                  depth=nlayers, gateDrop=dropout_rate)

    # original
    def forward_orig(self, x, tdrop=True):
        if self.embedding:
            x = self.embed(x)
        s, out =  0, []
        for i in range(self.context):
            o, s, sMetrics = self.rnnhighway(x[:, i], s, tdrop)
            out += [o]
        batch_size = x.size(0)
        x = torch.stack(out, 1).view(batch_size * self.context, -1)
        return self.unembed(x).view(batch_size, self.context, -1)

    # mine
    def forward(self, x, tdrop=True):

        if self.embedding:
            x = self.embed(x)  # (N, W, D)

        xs = []
        s = 0
        """ changed to dim 0 from dim 1 (35, 20, 200)"""
        for i in range(x.size(0)):
            """
            CHANGED i POSITION IMPORTANT !! (35, 20, 200)
            moved to dim 0 from dim 1
            """

            o, s, s_metrics = self.rnnhighway(x[i, :, :], s, tdrop)
            s_metrics = torch.cat(s_metrics, 1).unsqueeze(1)

            if self.lm:
                xs.append(s_metrics)
            else:
                xs.append(s_metrics)

        # if using hway2hway just return hidden states
        # returns a 2, 20, 400 hidden rep for each layer
        # print(xs[0].size()) = 1, 20, 800
        """same, changed to dim 0 instead of dim 1 for lm"""
        if self.lm:
            #print("Hello")
            #print(len(xs))
            #print(xs[0].size())
            y_all = torch.cat(xs, 0).view((x.size(0), x.size(1), -1))  # 40, 20, 800
            # print(y_all.size())
            # print(y_all.size())
        else:
            y_all = torch.cat(xs, 1)  # 40, 20, 800

        # assert y_all.size() == x.size()

        if self.att:
            # torch.Size([40, 20, 800])
            # torch.Size([40, 400])
            # torch.Size([20, 40])
            y = self.dropout(o)  # (N, len(Ks)*Co)
            y = self.fc1(y)  # (N, C)
            # print("y size")
            # print(y.size())
            return y_all, y
        else:
            return y_all, o


class RNNHighwayDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 max_len, trg_soi, nlayers=2, dropout_rate=0.2, attention=False, cuda=True):
        super(RNNHighwayDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi
        self.att = attention
        self.cuda = cuda
        self.trainable = True
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(self.hidden_dim)
        # DecoderCell(embed_dim, hidden_dim)

        self.decodercell = RHNContextCell(embed_dim, h=hidden_dim, depth=nlayers, gateDrop=dropout_rate)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_h, prev_s, target=None):
        '''
        enc_h  : B x S x 2*H
        prev_s : B x H
        '''

        if target is not None:
            target_len, batch_size = target.size(0), target.size(1)
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))
            if self.cuda:
                dec_h = dec_h.cuda()

            target = self.embed(target)
            for i in range(target_len):
                # all correct as expected
                # enc_h: torch.Size([20, 40, 800])
                # prev_s: torch.Size([20, 400])
                # ctx: torch.Size([20, 800])
                # target: torch.Size([40, 20, 400])
                ctx = self.attention(enc_h, prev_s)
                prev_s, s, sout = self.decodercell(target[i, :], prev_s, ctx, self.trainable)
                dec_h[:, i, :] = prev_s  # .unsqueeze(1)
            dec_h = dec_h.permute(1, 0, 2)
            outputs = self.dec2word(dec_h)
        # for prediction
        else:
            batch_size = enc_h.size(0)
            target = Variable(torch.LongTensor([self.trg_soi] * batch_size), volatile=True).view(batch_size, 1)
            outputs = Variable(torch.zeros(batch_size, self.max_len, self.vocab_size))
            if self.cuda:
                target = target.cuda()
                outputs = outputs.cuda()
            for i in range(self.max_len):
                target = self.embed(target).squeeze(1)
                ctx = self.attention(enc_h, prev_s)
                prev_s, s, sout = self.decodercell(target, prev_s, ctx, self.trainable)
                output = self.dec2word(prev_s)
                outputs[:, i, :] = output
                target = output.topk(1)[1]
        return outputs


class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim * 2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ctx_weights = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, trg_word, prev_s, ctx):
        '''
        trg_word : B x E
        prev_s   : B x H
        ctx      : B x 2*H
        '''
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx)
        reset_gate, update_gate = gates.chunk(2, 1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = torch.tanh(prev_s_tilde)
        prev_s = torch.mul((1 - reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s


if __name__ == "__main__":

    vocab_size = 10000
    hidden_size = 400
    batch_size = 20
    sent_len = 40
    nlayers = 2
    dropout_rate = 0.2
    rnn = True
    encode = False
    x = torch.randint(0, vocab_size, (batch_size, sent_len)).type(torch.LongTensor)

    if rnn:
        rnnhway_net = RecurrentHighwayText(vocab_size=vocab_size,
                               hidden_size=hidden_size,
                               dropout_rate=dropout_rate,
                               encode=encode)
        if encode:
            y = rnnhway_net(x)
            print(y[0].size())
            print(y[1].size())
        else:
            y, y_all = rnnhway_net(x)
            print(y_all.size())
    else:
        hway_net = HighwayText(vocab_size=vocab_size,
                             hidden_size=hidden_size,
                             sent_len=sent_len,
                             dropout_rate=dropout_rate
                            )
        y = hway_net(x)
        print(y.size())