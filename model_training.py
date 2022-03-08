import argparse, logging
import random, time, sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from build_dataset import get_cifar10_dataloader, build_search_spine3, build_search_Optimizer_Loss
from cell_archit import NetworkCIFAR

from utils import dagnode, Plot_network, create_dir, count_parameters_in_MB, Calculate_flops
# from utils import dagnode, create__dir, count_parameters_in_MB, Calculate_flops
import collections,utils

# import os
# os.environ["CUDA_VISIBLE_DEVICE"] = '1'


def Model_train(train_queue, model, train_criterion, optimizer, scheduler,
                args, valid_queue, eval_criterion, print_=False):

    since_time = time.time()

    global_step = 0
    total = len(train_queue)
    for epoch in range(args.search_epochs):
        objs = utils.AvgrageMeter() # loss
        top1 = utils.AvgrageMeter() # top1
        top5 = utils.AvgrageMeter()

        # switch to train mode
        model.train()

        batchtime = time.time()
        for step, (inputs, targets) in enumerate(train_queue):
            print('\r[Epoch:{0:>2d}/{1:>2d}, Training {2:>2d}/{3:>2d}, every step time {4:.2f}s, all used_time {5:.2f}min]'
                  .format(epoch+1, args.search_epochs, step+1, total, time.time()-batchtime, (time.time()-since_time)/60), end='')


            inputs, targets = inputs.to(args.device), targets.to(args.device) #

            optimizer.zero_grad()
            outputs = model(inputs, step=global_step)
            global_step += 1
            if args.search_use_aux_head:
                outputs, outputs_aux =outputs[0], outputs[1]

            loss = train_criterion(outputs, targets)
            if args.search_use_aux_head:
                loss_aux = train_criterion(outputs_aux, targets)
                loss += args.search_auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.search_grad_bound)
            optimizer.step() #

            # prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 2)) #
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5)) #
            objs.update(loss.data, inputs.size(0))
            top1.update(prec1.data, inputs.size(0))
            top5.update(prec5.data, inputs.size(0))

            batchtime = time.time()

        scheduler.step()

        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
            print('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))
            logging.info('train accuracy top1:{0:.3f}, train accuracy top5:{1:.3f}, train loss:{2:.5f}'.format(top1.avg,top5.avg,objs.avg))

            valid_top1_acc, valid_top5_acc, loss = Model_valid(valid_queue, model, eval_criterion, args)



    used_time = (time.time()-since_time) / 60

    return top1.avg, top5.avg, objs.avg, valid_top1_acc, valid_top5_acc, loss, used_time


def Model_valid(valid_queue, model, eval_criterion,args):

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

            if args.search_use_aux_head:
                outputs, outputs_aux =outputs[0], outputs[1]

            loss = eval_criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            # prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 2))
            objs.update(loss.data, inputs.size(0))
            top1.update(prec1.data, inputs.size(0))
            top5.update(prec5.data, inputs.size(0))

    print('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))
    logging.info('valid accuracy top1:{0:.3f}, valid accuracy top5:{1:.3f}, valid loss:{2:.5f}'.format(top1.avg, top5.avg, objs.avg))

    return top1.avg, top5.avg, objs.avg


def solution_evaluation(model, train_queue, valid_queue, args):
    num_parameters = count_parameters_in_MB(model)
    # ==================== build optimizer, loss and scheduler ====================
    # train_criterion, eval_criterion, optimizer, scheduler = build_search_Optimizer_Loss(model, args, epoch=-1)
    model.cuda()  # gpu
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    # SGD优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        0.01, # there is a doubt: why the last lr=0.001
        momentum=args.search_momentum,
        weight_decay=args.search_l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.search_epochs, args.search_lr_min, -1)
    # ==================== training the individual model and get valid accuracy ====================
    result = Model_train(train_queue, model, train_criterion, optimizer, scheduler, args, valid_queue, eval_criterion, print_=False) # True
    #
    Flops = Calculate_flops(model)

    # result[0]=valid_top1_acc, result[6]=used_time
    return 1-result[3]/100, num_parameters, Flops, result[6]


