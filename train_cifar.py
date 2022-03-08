# -*- coding: utf-8 -*-
import argparse, logging
import random, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import AEMO_Main as mm # 导入主函数函数
from build_dataset import build_search_spine3
from cell_archit import NetworkSpine, NetworkCIFAR
from utils import create_dir, count_parameters_in_MB, Calculate_flops, Plot_network
import utils
import shutil
from build_dataset import get_cifar10_dataloader, get_cifar100_dataloader, build_search_spine3, build_search_Optimizer_Loss


# @Reader : Labyrinthine Leo
# @Time   : 2020.12.24
# @role   : training the spine model searched by AEMONAS

def model_train(train_queue, model, train_criterion, optimizer, scheduler,
                args, valid_queue, eval_criterion, test_queue, print_=False):

    since_time = time.time()

    train_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    train_loss_list = []
    valid_loss_list = []

    global_step = 0
    best_prec1 = 0
    total = len(train_queue)
    for epoch in range(args.train_epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        # switch to train mode
        model.train()

        batchtime = time.time()
        for step, (inputs, targets) in enumerate(train_queue):
            print('\r[Epoch:{0:>2d}/{1:>2d}, Training {2:>2d}/{3:>2d}, every step time {4:.2f}s, all used_time {5:.2f}min]'
                  .format(epoch+1, args.train_epochs, step+1, total, time.time()-batchtime, (time.time()-since_time)/60), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs, step=global_step)
            global_step += 1
            if args.train_use_aux_head:
                outputs, outputs_aux = outputs[0], outputs[1]

            loss = train_criterion(outputs, targets)
            if args.train_use_aux_head:
                loss_aux = train_criterion(outputs_aux, targets)
                loss += args.train_auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_bound)
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 2))

            objs.update(loss.data, inputs.size(0))
            top1.update(prec1.data, inputs.size(0))
            top5.update(prec5.data, inputs.size(0))

            batchtime = time.time()

        train_acc_list.append(prec1)
        # train_loss_list.append(loss)

        scheduler.step()

        logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
        print('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))
        logging.info('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))

        valid_top1_acc, valid_top5_acc, loss = model_valid(valid_queue, model, eval_criterion, args)

        # acc1 = model_test(test_queue, model, eval_criterion)

        valid_acc_list.append(valid_top1_acc)
        # test_acc_list.append(acc1)
        # valid_loss_list.append(loss)

        # remember best prec1 and save checkpoint
        is_best = (valid_top1_acc > best_prec1)
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            # 'curr_prec1': prec1,
        }, is_best)



    used_time = (time.time()-since_time) / 60

    # return train_acc_list, valid_acc_list, test_acc_list
    return train_acc_list, valid_acc_list

def model_valid(valid_queue, model, eval_criterion, args):
    total = len(valid_queue) # the nums of batch

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (inputs, targets) in enumerate(valid_queue):
            print('\r[-------------Validating {0:>2d}/{1:>2d}]'.format(step+1, total), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            if args.train_use_aux_head:
                outputs, outputs_aux =outputs[0], outputs[1]

            loss = eval_criterion(outputs, targets)

            # prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            objs.update(loss.data, inputs.size(0))
            top1.update(prec1.data, inputs.size(0))
            top5.update(prec5.data, inputs.size(0))

    print('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))
    logging.info('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))

    return top1.avg, top5.avg, objs.avg

def model_test(test_loader, model, criterion):
    model.eval()
    acc1_sum, acc2_sum, n = 0.0, 0.0, 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for input, target in test_loader:
            # print(target.data.numpy())
            true_labels += list(target.data.numpy())
            input = input.cuda()
            target = target.cuda()

            input_var = input
            target_var = target

            # compute output
            output = model(input_var)
            # print(output.argmax(dim=1, keepdim=True).view(-1).cpu().numpy())
            pred_labels += list(output.argmax(dim=1, keepdim=True).view(-1).cpu().numpy())
            loss = criterion(output, target_var)

            # measure utils.accuracy and record loss
            prec1, prec2 = utils.accuracy(output.data, target, topk=(1, 5))
            # print("loss: {:.3f}".format(loss.cpu().numpy()), "acc1: {:.3f}".format(prec1.cpu().numpy()), "acc2: {:.3f}".format(prec2.cpu().numpy()))

            acc1_sum += prec1 * target.shape[0]
            acc2_sum += prec2 * target.shape[0]

            n += target.shape[0]

        acc1 = acc1_sum / n
        acc2 = acc2_sum / n

        print(' * Prec@1 {:.3f} Prec@2 {:.3f}'.format(acc1, acc2))

    return acc1

def build_cifar10_dataset(args):
    """
    Building the cifar dataset(10/100 classes), and get the train/valid queue
    :return: None
    """
    train_queue, valid_queue, test_queue = get_cifar10_dataloader(batch_size=args.train_batch_size, num_workers=args.num_work, shuffle=False)
    return train_queue, valid_queue, test_queue

def build_cifar100_dataset(args):
    """
    Building the cifar dataset(10/100 classes), and get the train/valid queue
    :return: None
    """
    train_queue, valid_queue, test_queue = get_cifar100_dataloader(batch_size=args.train_batch_size, num_workers=args.num_work, shuffle=False)
    return train_queue, valid_queue, test_queue


def save_checkpoint(args, state, is_best):

    filename = '{}/AEMONet_latest.pth.tar'.format(args.save)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        '{}/AEMONet_best.pth.tar'.format(args.save))


def main(args):
    x = [] # insert the best encoding

    solution = mm.Individual(x)

    Plot_network(solution.dag[0], '{}/best_conv_dag.png'.format(args.save))
    Plot_network(solution.dag[1], '{}/best_reduc_dag.png'.format(args.save))


    if args.dataset == 'cifar10':
        print('build cifar10 dataset')
        train_queue, valid_queue, test_queue = build_cifar10_dataset(args)  # get cifar dataset
    elif args.dataset == 'cifar100':
        train_queue, valid_queue, test_queue = build_cifar100_dataset(args)  # get cifar dataset


    # 构建模型
    model = NetworkCIFAR(args, args.classes, args.train_layers, args.train_channels, solution.dag, args.train_use_aux_head,
                         args.train_keep_prob, args.train_steps, args.train_drop_path_keep_prob,
                         args.train_channels_double)

    num_parameters = count_parameters_in_MB(model)
    print("Model Params: {} Mb".format(num_parameters))

    # ==================== build optimizer, loss and scheduler ====================
    # train_criterion, eval_criterion, optimizer, scheduler = build_search_Optimizer_Loss(model, args, epoch=-1)
    model.cuda()  # gpu
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        0.01,  # there is a doubt: why the last lr=0.001
        momentum=args.train_momentum,
        weight_decay=args.train_l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epochs, args.train_lr_min, -1)

    # ==================== training the individual model and get valid/test accuracy ====================

    result = model_train(train_queue, model, train_criterion, optimizer,
                         scheduler, args, valid_queue, eval_criterion, test_queue,
                         print_=True)  # True

    acc1 = model_test(test_queue, model, eval_criterion)
    print(acc1)

    res = np.vstack(result)
    logging.info(res)
    print(np.max(res, axis=1, keepdims=True))




if __name__=="__main__":

    # ===================================  args  ===================================
    # *******************  common setting  ******************
    parser = argparse.ArgumentParser(description='training on cifar dataset')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default='result')

    # ********************  dataset setting  ******************
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--train_autoaugment', action='store_true', default=False)
    parser.add_argument('--num_work', type=int, default=12, help='the number of the data worker.')

    # ****************** optimization setting  ******************
    parser.add_argument('--train_epochs', type=int, default=600)
    parser.add_argument('--train_lr_max', type=float, default=0.025)
    parser.add_argument('--train_lr_min', type=float, default=0.001)
    parser.add_argument('--train_momentum', type=float, default=0.9)
    parser.add_argument('--train_l2_reg', type=float, default=1e-5)
    parser.add_argument('--train_grad_bound', type=float, default=5.0)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=500)
    parser.add_argument('--train_steps', type=int, default=50000)

    # *********************  structure setting  ******************
    parser.add_argument('--train_use_aux_head', action='store_true', default=False)
    parser.add_argument('--train_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('--train_layers', type=int, default=1)
    parser.add_argument('--train_keep_prob', type=float, default=0.6)  # 0.6 also for final training
    parser.add_argument('--train_drop_path_keep_prob', type=float,
                        default=0.8)
    parser.add_argument('--train_channels', type=int, default=16)
    parser.add_argument('--train_channels_double', action='store_true',
                        default=True)  # False for Cifar, True for ImageNet model

    args = parser.parse_args()

    args.save = '{}/AEMO_train_{}_{}'.format(args.save, args.dataset, time.strftime("%Y-%m-%d-%H-%M-%S"))

    create_dir(args.save)

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    # ===================================  random seed setting  ===================================
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    # main
    main(args)
