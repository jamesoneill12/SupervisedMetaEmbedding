from models.autoencoders import *
from models.losses import CosineLoss, dice_loss


def get_loss(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cosine':
        criterion = CosineLoss()
    elif loss == 'kl':
        criterion = nn.KLDivLoss()
    elif loss == 'manhattan':
        criterion = nn.L1Loss()
    elif loss == 'nll':
        #criterion = Nlog_Likelihood()
        # REMEMBER !!!
        criterion = nn.NLLLoss()
    elif loss == 'dice':
        criterion = dice_loss
    return (criterion)


def get_model(args):
    if args.model == 'caeme':
        model = Autoencoder(args)
    elif args.model == 'daeme':
        model = DAutoencoder(args)
    return (model)

