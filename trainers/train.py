# coding: utf-8

import time
import os
import torch.onnx
from .helpers import save_vocab
from torch import nn
from models.rnn import RNNModel
from trainers.helpers import get_optimizer, get_scheduler
from .batchers import *


# target corpus is now added in the case of seq2seq
def run(args, corpus):
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: {}".format(device))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.pretrained is not None:
        args.emsize = 300
        args.nhid = 300

    train_data = corpus.train
    if 'sent' not in args.data:
        val_data = corpus.valid
    test_data = corpus.test
    ntokens = len(corpus.dictionary.word2idx)
    ntags = len(corpus.tag_vocab)

    print("Number of tags: {}".format(ntags))

    if 'pos' in args.data:
        task = 'pos'
    elif 'chunk' in args.data:
        task = 'chunking'
    elif 'ner' in args.data:
        task = 'ner'
    elif 'sent' in args.data:
        task = 'sentiment'

    model = RNNModel(rnn_type=args.model, ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, bsize=args.batch_size,
                     nout=ntags, nlayers=args.nlayers, pretrained=args.pretrained, task=task,
                     drop_rate=args.dropout, tie_weights=args.tied, tune_weights=args.tunable).to(device)

    if "sent" in args.data:
        criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optim = get_optimizer(model, args)
    # if None, do the original annealing from lr = 20.
    scheduler = get_scheduler(optim, args, train_data)

    ###############################################################################
    # Training code
    ###############################################################################

    performance = {'train_epoch': [], 'train_loss': [], 'train_acc': [], 'train_lr': [],
                   'val_epoch': [], 'val_loss': [], 'val_acc': [], 'val_lr': [],
                   'test_loss': [], 'test_acc': []}

    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_acc = 0.
        ntokens = len(corpus.dictionary.idx2word)
        # was eval batch size but changed because thought it was causing error
        hidden = model.init_hidden(args.batch_size)
        with torch.no_grad():
            for i, batch in enumerate(iter(data_source)):
                data, targets = batch.text, batch.label
                if data.size(1) != args.batch_size:
                    pad_tensor = torch.zeros((data.size(0), args.batch_size - data.size(1))).type(
                        torch.cuda.LongTensor)
                    data = torch.cat([data, pad_tensor], 1)
                    targets = torch.cat([targets, pad_tensor], 1)

                output, hidden = model(data, hidden)

                if 'sent' not in args.data:
                    output = output.view(-1, output.size(2))

                targets = targets.view(-1, )

                loss = criterion(output, targets)
                _, predicted = torch.max(output.data, 1)
                hidden = repackage_hidden(hidden)
                total_loss += len(data) * loss.item()

                total_acc += (predicted == targets).sum().item()/(len(predicted))

        val_acc = 100 * (total_acc / len(data_source))
        val_loss = total_loss / len(data_source)
        if args.scheduler is not None:
            scheduler.step(val_loss)
        return val_loss, val_acc

    def train(train_perc=None):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        total_correct = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary.idx2word)
        hidden = model.init_hidden(args.batch_size)
        for i, batch in enumerate(iter(train_data)):
            x_train, targets = batch.text, batch.label
            # print(x_train.size())
            # print(targets.size())
            if x_train.size(1) != args.batch_size:
                pad_tensor = torch.zeros((x_train.size(0), args.batch_size - x_train.size(1))).type(torch.cuda.LongTensor)
                x_train = torch.cat([x_train, pad_tensor], 1)
                if 'sent' in args.data:
                    print(targets.size())
                    print(pad_tensor.size())
                    targets = torch.cat([targets, pad_tensor], 0)
                else:
                    targets = torch.cat([targets, pad_tensor], 1)
            hidden = repackage_hidden(hidden)
            model.zero_grad()

            output, hidden = model(x_train, hidden)
            # backward wont work for ptb_/output.permute(0,2,1).contiguous() because shouldn't contiguous()

            # print("First")
            # print(output.size())
            # print(targets.size())

            if 'sent' not in args.data:
                output = output.view(-1, output.size(2))
            targets = targets.view(-1, )

            # print("Second")
            # print(output.size())
            # print(targets.size())
            loss = criterion(output, targets)

            if args.optimizer is not None:
                optim.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if args.optimizer is not None:
                optim.step()
            else:
                if args.pretrained is not None:
                    for name, p in model.named_parameters():
                        if name is not 'encoder.weight':
                            p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()
            output_flat = output.view(-1, ntags)
            _, predicted = torch.max(output_flat.data, 1)
            total_correct += ((predicted == targets).sum().item() / float(len(predicted))) * 100

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / (args.log_interval +1)
                cur_acc = total_correct / (args.log_interval + 1)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | acc {:8.2f}'.format(epoch, i, len(train_data) // args.bptt, lr,
                                                          elapsed * 1000 / args.log_interval, cur_loss, cur_acc))
                performance['train_epoch'].append(epoch)
                performance['train_lr'].append(lr)
                performance['train_loss'].append(cur_loss)
                performance['train_acc'].append(cur_acc)
                total_loss = 0
                total_correct = 0
                start_time = time.time()

    def export_onnx(path, batch_size, seq_len):

        print('The model is also exported in ONNX format at {}'.
              format(os.path.realpath(args.onnx_export)))
        model.eval()
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
        hidden = model.init_hidden(batch_size)
        torch.onnx.export(model, (dummy_input, hidden), path)

    # Loop over epochs.
    lr = args.lr  # if args.optimizer == None:
    best_val_loss = None
    anneal_inc = 0
    anneal_dec = False

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_perc = epoch / float(args.epochs)
            train(train_perc)
            if 'sent' not in args.data:
                val_loss, val_acc = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, val_acc))
                print('-' * 89)
                performance['val_epoch'].append(epoch)
                performance['val_lr'].append(lr)
                performance['val_loss'].append(val_loss)
                performance['val_acc'].append(val_acc)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Run on test data.
    test_loss, test_acc = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test acc {:8.2f}'.format(test_loss, test_acc))
    print('=' * 89)
    performance['test_loss'].append(test_loss)
    performance['test_acc'].append(test_acc)

    save_vocab(performance, args.results_path, show_len=False)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)