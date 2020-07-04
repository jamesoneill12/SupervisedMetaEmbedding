import torch
from torch.nn import Parameter
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


"""module, weights, dropout=0, variational=False"""


def dropconnect(module, weights, p=None, dim=None, method='standard'):
    """module, weights, dropout=0, variational=False"""
    if method == 'standard':
        return DropConnect(module, weights, p, method)
    elif method == 'gaussian':
        return GaussianDropConnect(p/(1-p))
    elif method == 'variational':
        return VariationalDropConnect(p/(1-p), dim)
    elif method == 'concrete':
        # takes  layer, input_shape
        return ConcreteDropConnect


class DropConnect(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, method='standard'):
        super(DropConnect, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.method = method
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.method == 'variational':
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            elif self.method == 'concrete':
                pass
            elif self.method == 'locked':
                pass
            elif self.method == 'curriculum':
                pass
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)



class LockedDropConnect(nn.Module):
    """https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb"""
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class GaussianDropConnect(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropConnect, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class VariationalDropConnect(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropConnect, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


class ConcreteDropConnect(nn.Module):
    def __init__(self, layer, input_shape, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropConnect, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        # Check if batch size is the same as unif_noise, if not take care
        unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size())))).cuda()

        drop_prob = (torch.log(self.p + eps)
                     - torch.log(1 - self.p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square


class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class CurriculumDropConnect(nn.Module):
    """
    :param
    gamma : temperature I think ??
    p : scheduled probability throughout training, reust ss_prob func
    """
    def __init__(self):
        super(CurriculumDropConnect, self).__init__()

    def forward(self, x, gamma, p):
        return (1.-p) * np.exp(-gamma*x) + p


def get_dropout(drop_position, drop_rate, drop_dim, drop_method, fixed_dropout=False):
    if drop_position == 1 or drop_position == 3:
        drop_in = dropout(drop_rate, drop_dim,
                          drop_method, fixed=fixed_dropout)
    else:
        drop_in = False
    if drop_position == 2 or drop_position == 3:
        drop_out = dropout(drop_rate, drop_dim,
                           drop_method, fixed=fixed_dropout)
    else:
        drop_out = False
    return drop_in, drop_out


def dropout(p=None, dim=None, method='standard', fixed=False):
    if method == 'standard':
        return LockedDropout(p) if fixed else nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p/(1-p), fixed=fixed)
    elif method == 'locked':
        return LockedDropout(p)
    elif method == 'variational':
        """This is specifically gaussian variational dropout 
        and doesn't converge for either fixed time steps or non-fixed"""
        return VariationalDropout(p/(1-p), dim, locked=fixed)
    elif method == 'concrete':
        # takes  layer, input_shape
        return ConcreteDropout
    # elif method == 'zoneout':
        # return Zoneout(p, fixed)
    elif method == 'curriculum':
        """Not required, can just change nn.Dropout() param p"""
        # return CurriculumDropout()
        return nn.Dropout(p)


class LockedDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.p = dropout

    def forward(self, x, p=None):
        if p is not None:
            self.p = p
        if not self.training or not p:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        mask = Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0, fixed=False):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        self.fixed = fixed

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            if self.fixed:
                epsilon = torch.randn((1, x.size(1), x.size(2))) * self.alpha + 1
            else:
                epsilon = torch.randn(x.size()) * self.alpha + 1
            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class VariationalDropout(nn.Module):
    """
    Variational Gaussian Dropout is not Bayesian so read this paper:
    https://arxiv.org/abs/1711.02989
    """

    # max alpha is used for clamping and should be small
    def __init__(self, alpha=0.01, dim=None, locked=True):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        self.locked = locked
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        alpha = self.log_alpha.exp()
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3
        kl = -negative_kl
        return kl.mean()

    def forward(self, x):
        """Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e"""
        if self.train():
            # N(0,1)

            if self.locked:
                epsilon = Variable(torch.randn(size=(x.size(0), x.size(2))))
                epsilon = torch.cat([epsilon] * x.size(1)).view(x.size())
            else:
                epsilon = Variable(torch.randn(x.size()))

            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


"""https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb"""


class ConcreteDropout(nn.Module):
    def __init__(self, layer, input_shape, weight_regularizer=1e-6, locked = True,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.locked = locked
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        # Check if batch size is the same as unif_noise, if not take care
        if self.locked:
            noise = np.random.uniform(size=(x.size(0), x.size(2)))
            noise = np.repeat(noise[:, np.newaxis, :], x.size(1), axis=1)
        else:
            noise = np.random.uniform(size=tuple(x.size()))

        unif_noise = Variable(torch.FloatTensor(noise)).cuda()

        drop_prob = (torch.log(self.p + eps)
                     - torch.log(1 - self.p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square


class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class CurriculumDropout(nn.Module):
    """
    :param
    gamma : temperature
    p : scheduled probability throughout training, reust ss_prob func
    """
    def __init__(self):
        super(CurriculumDropout, self).__init__()

    def forward(self, x, gamma, p):
        if self.train():
            return (1.-p) * np.exp(-gamma * x) + p
        else:
            return x


def show_drop_probs(model, dropout_position):
    if dropout_position == 1:
        print("drop-in {}".format(model.drop_in.p))
    elif dropout_position == 2:
        print("drop-out {}".format(model.drop_out.p))
    elif dropout_position == 3:
        print("drop-in {} \t drop-out {}".format(model.drop_in.p, model.drop_out.p))