from models.highway import RecurrentHighwayText
from models.regularizers import *
from embeddings.get_embeddings import load_meta_embedding


def get_rnn(rnn_type, ninp, nhid, nlayers, dropout, drop_method = 'standard'):
    if rnn_type in ['LSTM', 'GRU']:
        rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
    elif rnn_type == 'HIGHWAY':
        rnn = RecurrentHighwayText(ninp, nhid, nlayers, drop_method=drop_method, lm=True,
                                   dropout_rate=dropout, embedding=False, attention=False)
    else:
        try:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        except KeyError:
            raise ValueError("""An invalid option for `--model` was supplied,
                             options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
    return rnn


class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nout, nlayers, bsize, pretrained, task=None,
                 drop_rate=0.2, drop_method='standard', drop_position=1, fixed_drop=True,
                 batch_norm=False, tie_weights=False, tune_weights=False):
        super(RNNModel, self).__init__()

        drop_dim = nhid if drop_method == 'variational' else None

        self.batch_norm = batch_norm
        self.task = task
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers
        self.drop_method = drop_method
        self.drop_position = drop_position
        self.batch_size = bsize
        self.reg_loss = 0
        self.fixed_dropout = fixed_drop

        if self.batch_norm:
            self.bnorm_in = nn.BatchNorm1d(nhid)
            self.bnorm_out = nn.BatchNorm1d(nhid)

        self.rnn = get_rnn(rnn_type, ninp, nhid,
                           nlayers, drop_rate, drop_method)

        if pretrained == "all":
            emb_size = 1800
        elif pretrained is not None:
            emb_size = 300

        if pretrained is not None:
            # task="chunking", emb_choice="all", tune_weihts = False
            self.encoder = nn.Sequential(load_meta_embedding(task, pretrained, tune_weights),
                                         nn.Linear(emb_size, nhid))
            self.decoder = nn.Linear(nhid, nout)
            self.init_decoder_weights()
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
            print("Vocab Shape {}".format(self.encoder.weight.size()))

            if self.rnn_type.lower() == "highway":
                self.decoder = nn.Linear(nhid * 2, nout)
            else:
                self.decoder = nn.Linear(nhid, nout)
            self.init_weights()

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        "module, weights, drop_position, drop_rate, drop_dim, drop_method"
        # weight_ih_l, 'weight_hh_l{}{} 'bias_ih_l{}{}', 'bias_hh_l{}{}'
        rnn_param_names = [name for name, _ in self.rnn.named_parameters()]
        encoder_param_names = [name for name, _ in self.encoder.named_parameters()]
        decoder_param_names = [name for name, _ in self.decoder.named_parameters()]

        """Takes care of activation dropout on input and output"""
        self.drop_in, self.drop_out = get_dropout(drop_position,
                                                  drop_rate, drop_dim,
                                                  drop_method, fixed_drop)

        """Takes care of activation dropout on input and output if using concrete"""
        if self.drop_method == 'concrete':
            if self.drop_in is not False:
                self.drop_in = self.drop_in(self.rnn, input_shape=(bsize, nhid),
                                            weight_regularizer=1e-6,
                                            dropout_regularizer=1e-5, locked=fixed_drop)
            if self.drop_out is not False:
                self.drop_out = self.drop_out(self.decoder, input_shape=(bsize, ntoken),
                                              weight_regularizer=1e-6,
                                              dropout_regularizer=1e-5, locked=fixed_drop)

    def init_encoder_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, 0.1)

    def init_decoder_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, 0.1)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_input_mask(self, emb, p=None):
        """assigned input dropout, unless concrete in which
        case its already wrapped as apart of self.rnn"""
        if self.drop_position in [1, 3]:
            if self.drop_method == 'curriculum' and p is not None:
                self.drop_in.p = p
                # emb = self.drop_in(emb, p)
            elif self.drop_method != 'concrete':
                emb = self.drop_in(emb)
        return emb

    def get_output_mask(self, output, p = None):
        """assigned output dropout, unless concrete in which
        case its already wrapped as apart of self.decoder"""
        if self.drop_position in [2, 3]:
            if self.drop_method == 'curriculum' and p is not None:
                self.drop_out.p = p
            if self.drop_method != 'concrete':
                output = self.drop_out(output)
        return output

    def forward_pass(self, emb, hidden):
        """
        Takes care of the mess for highway networks and concrete
        i.e when using concrete, rnn already wrapped and when using highway,
        only need to pass emb and not hidden
        """

        if self.drop_position in [1, 3]:
            if self.rnn_type.lower() == "highway" and self.drop_method != 'concrete':
                output, hidden = self.rnn(emb)
            elif self.rnn_type.lower() == "highway" and self.drop_method == 'concrete':
                output, hidden = self.drop_in(emb)
                self.reg_loss += self.drop_in.regularisation()
            else:
                output, hidden = self.rnn(emb, hidden)
        else:
            if self.rnn_type.lower() == "highway":
                output, hidden = self.rnn(emb)
            else:
                output, hidden = self.rnn(emb, hidden)
        return output, hidden

    """ gamma and p are only needed when using curriculum dropout I think. """
    """ CORRECTED: Just need to update nn.Dropout().p"""
    def forward(self, x, hidden, p=None, target=None):

        self.reg_loss = 0
        emb = self.encoder(x)
        emb = self.get_input_mask(emb, p)

        if self.batch_norm:
            emb = self.bnorm_in(emb)

        output, hidden = self.forward_pass(emb, hidden)

        if self.batch_norm:
            output = self.bnorm_out(output)

        output = self.get_output_mask(output, p)

        if self.task == "sentiment":
            output = output[-1, :, :]
            # print(output.size())
            decoded = self.decoder(output)
            # print(decoded.size())
            return decoded, hidden

        if self.drop_position in [2, 3]:
            if self.rnn_type.lower() == 'highway' and self.drop_method != 'concrete':
                decoded = self.decoder(output.view(output.size(0) *
                                                   output.size(1), output.size(2)))
            elif self.drop_method == 'concrete':
                """no self.decoder since it is apart of self.drop_out when concrete used.
                Also haven't flattened dim 0 and 1 like below here """
                decoded = self.drop_out(output)
                self.reg_loss += self.drop_out.regularisation()
                # print("second")
                return decoded, hidden
            else:
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        else:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def regularisation_loss(self, clear=False):
        reg_loss = self.forward_main[0].regularisation()\
                   +self.forward_main[1].regularisation()\
                   +self.forward_main[2].regularisation()
        return reg_loss
