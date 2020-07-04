import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class CosineLoss(torch.nn.Module):
    """
    Cosine loss function.
    Minimizes reconstruction loss with respect to angles
    """

    # was 2.0 before using the sigmoid
    def __init__(self, margin=2.0, double_margin=False):
        super(CosineLoss, self).__init__()
        self.cosine = nn.MSELoss(size_average=False)

    def forward(self, pred, label):
        cosine_sim = F.cosine_similarity (pred, label)
        cosine_loss = torch.mean(torch.pow(1-cosine_sim, 2))
        return cosine_loss


class NLL(torch.nn.Module):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, input, target):
        _assert_no_grad(target)
        b = target * torch.log(input)
        a = (1-target) * torch.log(1 - input)
        loss = torch.mean(a-b)
        return loss
