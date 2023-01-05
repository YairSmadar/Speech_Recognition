import copy
import os
import time

import config
import torch
import torch.nn as nn
import models_factory
import data_loader
from sklearn.metrics import confusion_matrix

# import data_loader_built_in

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
best_prec1 = 0
best_state_dict = None

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_freq = opt.print_freq

    # switch to train mode
    model.train()

    for i, (_input, _target) in enumerate(train_loader):

        if i % print_freq == 0:
            time_start = time.time()
        output = model(_input)

        if device_name == 'cuda':
            _target = _target.cuda(non_blocking=True)

        loss = criterion(output, _target)

        prec1, prec5 = accuracy(output.data, _target, topk=(1, 5))

        losses.update(loss.item(), _input.size(0))
        top1.update(prec1.item(), _input.size(0))
        top5.update(prec5.item(), _input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            batch_time.update(time.time() - time_start)
            print('Train:\t[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(
        'Train:\t[{0}]\tLoss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'.format(epoch, loss=losses,
                                                                                                    top1=top1,
                                                                                                    top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_freq = opt.print_freq

    # switch to evaluate mode
    model.eval()

    # for confusion matrix
    y_pred = []
    y_true = []

    for i, (_input, _target) in enumerate(val_loader):

        if i % print_freq == 0:
            time_start = time.time()

        if device_name == 'cuda':
            _target = _target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(_input)

            output_for_conf = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output_for_conf)

            labels_for_conf = _target.data.cpu().numpy()
            y_true.extend(labels_for_conf)

            loss = criterion(output, _target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, _target, topk=(1, 5))
            losses.update(loss.item(), _input.size(0))
            top1.update(prec1.item(), _input.size(0))
            top5.update(prec5.item(), _input.size(0))

            if i % print_freq == 0:
                batch_time.update(time.time() - time_start)

                print('Test:\t[{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    cf_matrix = confusion_matrix(y_true, y_pred)
    # print(cf_matrix)

    print('Test:\t[{0}]\tLoss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'.format(epoch, loss=losses,
                                                                                                     top1=top1,
                                                                                                     top5=top5))
    if epoch % 5 == 0:
        from sklearn.metrics import precision_recall_fscore_support as score

        predicted = [1, 2, 3, 4, 5, 1, 2, 1, 1, 4, 5]
        y_test = [1, 2, 3, 4, 5, 1, 2, 1, 1, 4, 1]

        precision, recall, fscore, support = score(y_test, predicted)

        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        # print('fscore: {}'.format(fscore))
        # print('support: {}'.format(support))

    return losses.avg, top1.avg, top5.avg


def save_best_prec1(model, test_prcition1: list):
    global best_prec1
    global best_state_dict

    if test_prcition1[-1] > best_prec1:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_prec1 = test_prcition1[-1]

    return best_state_dict


def save_checkpoint(model, optimizer, epoch, opt,
                    train_losses, train_prcition1, train_prcition5, test_losses, test_prcition1, test_prcition5):
    # save best prec@1
    save_best_prec1(model, test_prcition1)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    model_type = 'LSTM' if opt.LSTM else 'CNN'
    full_path = os.path.join(opt.save_path, f'{model_type}_model.tar')

    # save checkpoint
    torch.save({'epoch': epoch,
                'model': model,
                'Best_state_dict': best_state_dict,
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_prcition1': train_prcition1,
                'train_prcition5': train_prcition5,
                'test_losses': test_losses,
                'test_prcition1': test_prcition1,
                'test_prcition5': test_prcition5,
                },
               full_path
               )


def main(opt):
    # initialize the results arrays
    train_prcition1, train_prcition5 = [], []
    test_prcition1, test_prcition5 = [], []
    train_losses, test_losses = [], []

    model = models_factory.get_model(opt)
    model = nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    if opt.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # get data
    data = data_loader.SpectogramDataset(opt, set_type='train')
    train_loader = data_loader.getDataLoader(data, opt.batch_size)

    data = data_loader.SpectogramDataset(opt, set_type='val')
    val_loader = data_loader.getDataLoader(data, opt.batch_size)
    val_loader.dataset.is_train = False

    # get data2
    # train_loader2 = data_loader_built_in.train_loader
    # test_loader2 = data_loader_built_in.test_loader

    for epoch in range(opt.epochs):
        train_loss, train_prc1, train_prc5 = train(train_loader, model, criterion, optimizer, epoch, opt)
        test_loss, test_prc1, test_prc5 = validate(val_loader, model, criterion, epoch, opt)

        if opt.use_scheduler:
            scheduler.step()

        # save results
        train_losses.append(train_loss)
        train_prcition1.append(train_prc1)
        train_prcition5.append(train_prc5)
        test_losses.append(test_loss)
        test_prcition1.append(test_prc1)
        test_prcition5.append(test_prc5)

        save_checkpoint(model, optimizer, epoch, opt,
                        train_losses, train_prcition1, train_prcition5, test_losses, test_prcition1, test_prcition5)


if __name__ == '__main__':
    parser = config.get_arguments()
    opt = parser.parse_args()

    main(opt)
