from __future__ import print_function

import matplotlib.pyplot as plot
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from utils import *
import sys
import os

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []


def adjust_learning_rate(optimizer, args, epoch, best_pred):
    lr = args.lr * (0.1 ** ((epoch - 1) // args.lr_decay))
    if (epoch-1) % args.lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init the args
    global best_pred, errlist_train, errlist_val
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    # plot 
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    dataset = importlib.import_module('dataloader.'+args.dataset)
    Dataloder = dataset.Dataloder
    classes, train_loader, test_loader = Dataloder(args).getloader()
    # init the model
    models = importlib.import_module('model.'+ 'DEPnet_V2')
    model = models.Net(len(classes))
    print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = torch.nn.DataParallel(model)

    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] +1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no resume checkpoint found at '{}'".\
                format(args.resume))

    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0,0,0
        adjust_learning_rate(optimizer, args, epoch, best_pred)
        for batch_idx, (data, target) in enumerate(train_loader):

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.data.max(1)[1]
            correct_ = pred.eq(target.data).cpu().sum().item()
            correct += correct_
            total_ = target.size(0)
            total += total_
            err = 100-100.*correct/total
            print('Epoch: {}, Iteration: {}, Batch_Accuracy: {}, Accuracy_Rate: {}'.format(epoch, batch_idx, correct_/total_, correct/total))

        errlist_train += [err]

    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0,0,0
        is_best = False
        for batch_idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct_ = pred.eq(target.data).cpu().sum().item()
            correct += correct_
            total_ = target.size(0)
            total += total_
            err = 100-100.*correct/total


        print(' ##### Testing Epoch: {}, Accuracy_Rate: {}'.format(epoch, correct/total))

        if args.eval:
            print('Error rate is %.3f'%err)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err 
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train':errlist_train,
            'errlist_val':errlist_val,
            }, args=args, is_best=is_best)
            

    if args.eval:
        test(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('Epoch:', epoch)
        train(epoch)
        test(epoch)

        # save train_val curve to a file
        if args.plot:
            plot.clf()
            plot.xlabel('Epoches: ')
            plot.ylabel('Error Rate: %')
            plot.plot(errlist_train, label='train')
            plot.plot(errlist_val, label='val')
            plot.legend(loc='upper left')
            plot.savefig("runs/%s/%s/%s"%(args.dataset, args.model, args.checkname)
                                +'train_val.jpg')
            plot.draw()
            plot.pause(0.001)

if __name__ == "__main__":
    main()
