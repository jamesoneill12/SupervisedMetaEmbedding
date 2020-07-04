import torch.nn as nn
import torch
import torch.nn.functional as F

"""
# not sure what this one was used for so be careful

class Attention(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, embedded, hidden, encoder_outputs):
        # concat the hidden vector and input vector
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # attention vector and concatenate with target word (embedded[0])
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights
"""

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, enc_h, prev_s):
        '''
        enc_h  : B x S x 2*H
        prev_s : B x 1 x H
        '''
        seq_len = enc_h.size(1)

        enc_h_in = self.enc_h_in(enc_h)  # B x S x H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B x 1 x H

        h = torch.tanh(enc_h_in + prev_s.expand_as(enc_h_in))  # B x S x H
        h = self.linear(h)  # B x S x 1

        alpha = torch.softmax(h, dim=1)
        ctx = torch.bmm(alpha.transpose(2, 1), enc_h).squeeze(1)  # B x 1 x 2*H

        return ctx