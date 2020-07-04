import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def dim_fixer(word_dims, fixed_dim):
    if word_dims < 101:
        if fixed_dim != None:
            hidden_dim = fixed_dim
        else:
            fixed_dim = 50
    else:
        if fixed_dim != None:
            hidden_dim = fixed_dim
        else:
            fixed_dim = 100
    return (hidden_dim)


def choose_activations(activations):
    if activations == 'sigmoid': return nn.Sigmoid()
    elif activations == 'relu': return nn.ReLU()
    elif activations == 'leaky': return nn.LeakyReLU()
    else: return nn.Tanh()


class Autoencoder(nn.Module):
    ''' ----------------- Standard (Stacked) AutoEncoder ---------------- '''
    def __init__(self, args):
        super(Autoencoder, self).__init__()

        self.num_layers = args.num_layers
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.dist = args.dist

        if args.model in ['daeme','caeme','aaeme']:
            self.hidden_dim = args.hidden_dim
        else:
            self.hidden_dim = args.hidden_dim if args.encode_relational_vectors \
                else dim_fixer(self.input_dims, args.hidden_dim)

        if args.multi_task:
            self.input_dim, self.output_dim, self.hidden_dim = 1700, 1700, 200
            self.fc3 = nn.Linear(self.hidden_dim, 100)
            self.fc4 = nn.Linear(100, 10)
            self.fc5 = nn.Linear(20, 1)
            self.mt_activation = choose_activations('sigmoid')

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = choose_activations(args.activations)
        self.loss = args.loss

        if args.loss in ['kl']: self.log_softmax = nn.LogSoftmax()

    def get_embedding(self, x):
        return (self.activation(self.fc1(x)))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.log_softmax(x) if self.loss == 'kl' else x
        return (x)

    def forward_multi(self, x1, x2):
        ys = []
        for x_s in [x1,x2]:
            x = self.activation(self.fc1(x_s))
            x = self.activation(self.fc3(x))
            x = self.activation(self.fc4(x))
            ys.append(x)
        if self.dist == 'linear':
            y = torch.cat((ys), 1)
            y = self.mt_activation(self.fc5(y))
        elif self.dist == 'euclidean':
            y = torch.exp(-F.pairwise_distance(ys[0],ys[1]))
        elif self.dist == 'cosine':
            y = torch.exp(-F.cosine_similarity(ys[0], ys[1]))
        return (y)


class DAutoencoder(nn.Module):
    def __init__(self, args):
        super(DAutoencoder, self).__init__()

        self.num_layers = args.num_layers
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.multi_task = args.multi_task

        if args.model in ['daeme', 'caeme', 'aaeme']:
            self.hidden_dim = args.hidden_dim
        else:
            self.hidden_dim = args.hidden_dim if args.encode_relational_vectors \
                else dim_fixer(self.input_dims, args.hidden_dim)

        self.activation = choose_activations(args.activations)
        self.loss = args.loss

        # left and right for decoupling
        if args.multi_task:
            if args.model == 'daeme':
                input_dim, hidden_dim = 850, 100
            else:
                input_dim, hidden_dim = 1700, 200

            self.fc1_right = nn.Linear(input_dim, hidden_dim)
            self.fc1_left = nn.Linear(input_dim, hidden_dim)
            self.fc2_left = nn.Linear(hidden_dim, input_dim)
            self.fc2_right = nn.Linear(hidden_dim, input_dim)
            self.fc3 = nn.Linear(hidden_dim, 50)
            self.fc4 = nn.Linear(50, 10)
            self.fc5 = nn.Linear(20, 1)
            self.mt_activation = choose_activations('sigmoid')

        else:
            self.fc1_right = nn.Linear(self.input_dim, int(self.hidden_dim / 2))
            self.fc1_left = nn.Linear(self.input_dim, int(self.hidden_dim / 2))
            self.fc2_left = nn.Linear(int(self.hidden_dim / 2), int(self.output_dim / 2))
            self.fc2_right = nn.Linear(int(self.hidden_dim / 2), int(self.output_dim / 2))

        if args.loss in ['kl']:
            self.log_softmax = nn.LogSoftmax()

    def get_embedding(self, x, side='left'):
        if self.multi_task:
            if side == 'left':
                return (self.activation(self.fc1_left(x)))
            else:
                return (self.activation(self.fc1_right(x)))
        else:
            x1 = self.activation(self.fc1_left(x))
            x2 = self.activation(self.fc1_right(x))
            x = torch.cat((x1, x2), 1)
        return (x)

    def forward(self, x,side='left'):
        if side=='left':
            x_left = self.activation(self.fc1_left(x))
            x = self.activation(self.fc2_left(x_left))
        else:
            x_right = self.activation(self.fc1_right(x))
            x = self.activation(self.fc2_right(x_right))

        # if uncommented, change last x to x_left/right
        #x = torch.cat((x_left, x_right), 1)
        x = self.log_softmax(x) if self.loss == 'kl' else x

        return (x)

    def forward_multi(self,x1,x2):
        ys = []
        for i, x_s in enumerate([x1,x2]):
            if i == 0:
                x_mt = self.activation(self.fc1_left(x_s))
            else:
                x_mt = self.activation(self.fc1_right(x_s))
            x_mt = self.activation(self.fc3(x_mt))
            x_mt = self.activation(self.fc4(x_mt))
            ys.append(x_mt)
        if self.dist == 'linear':
            y = torch.cat((ys), 1)
            y = self.mt_activation(self.fc5(y))
        elif self.dist == 'euclidean':
            y = torch.exp(-F.pairwise_distance(ys[0],ys[1]))
        elif self.dist == 'cosine':
            y = torch.exp(-F.cosine_similarity(ys[0], ys[1]))
        return (y)


class Stacked_Autoencoder(nn.Module):
    def __init__(self, args):
        super(Stacked_Autoencoder, self).__init__()

        self.num_layers = args.num_layers
        self.word_dims = args.word_dims
        self.hidden_dim = dim_fixer(self.word_dims, args.fixed_dim)
        self.dropout = args.dropout

        if self.dropout:
            self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(args.word_dims, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim * 2, args.word_dims)
        self.activation = choose_activations(args.activations)

    def forward_once(self, x, activation):
        x = self.activation(activation(x))
        if self.drop:
            x = self.drop(x)
        return (x)

    def get_embedding(self, x):
        x = self.forward_once(x, self.fc1)
        x = self.forward_once(x, self.fc2)
        x = self.forward_once(x, self.fc3)
        return (x)

    def forward(self, x):
        x = self.forward_once(x, self.fc1)
        x = self.forward_once(x, self.fc2)
        x = self.forward_once(x, self.fc3)
        x = self.forward_once(x, self.fc4)
        return x




class VAE(nn.Module):
    ''' ---------- Variational AutoEncoder ------------------ '''
    def __init__(self, args):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(args.word_dims, 100)
        self.fc21 = nn.Linear(100, 100)
        self.fc22 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, args.word_dims)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[0]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


####### ---------------------------------- Adversarial AutoEncoder ----------------------------------- #############

# Encoder
class Q_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.droppout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.droppout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self, X_dim, N, z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

