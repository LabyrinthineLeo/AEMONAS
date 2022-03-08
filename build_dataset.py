import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dataset

from autoaugment import CIFAR10Policy
from auto_augment import AutoAugment
import numpy as np
import argparse
import torch.utils.data as Data # 数据加载器
import torchvision.datasets as datasets # benchmark datasets

# ============================== dataset ==================================
# == autoaugment:       https://github.com/DeepVoltaire/AutoAugment      ==
# == auto_augment:      https://github.com/4uiiurz1/pytorch-auto-augment ==
# =========================================================================

class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(cutout_size, autoaugment=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    else: # 不做自动增广
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))


    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def build_search_cifar10_(args, ratio=0.9, cutout_size=None, autoaugment=False, num_workers = 10):


    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)


    train_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.search_train_batch_size, # 128
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=num_workers
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.search_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=num_workers
    )

    return train_queue, valid_queue

def get_cifar10_dataloader(batch_size, num_workers, shuffle, resize=None):
    """
    :param batch_size: dataloader batchsize
    :param num_workers: dataloader num_works
    :param shuffle: flag of shuffle
    :param resize: resize
    :return: train dataloader & valid dataloader
    """

    # data transform
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    ratio = 0.9

    train_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    valid_trans = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    if resize:
        train_trans.insert(1, transforms.Resize(size=resize))
        valid_trans.insert(0, transforms.Resize(size=resize))


    train_data = datasets.CIFAR10(
        root='./data/CIFAR10',
        train=True,
        transform=transforms.Compose(train_trans),
        download=False
    )

    valid_data = datasets.CIFAR10(
        root='./data/CIFAR10',
        train=True,
        transform=transforms.Compose(valid_trans),
        download=False
    )

    test_data = datasets.CIFAR10(
        root='./data/CIFAR10',
        train=False,
        transform=transforms.Compose(valid_trans),
        download=False
    )

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))
    split = int(np.floor(ratio * num_train))
    print('split ratio:{}, train nums:{}, valid nums:{}, test nums:{}'.format(ratio, split, num_train-split, len(test_data)))

    np.random.shuffle(indices)

    train_loader = Data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        sampler = Data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers = num_workers
    )

    valid_loader = Data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        sampler=Data.sampler.SubsetRandomSampler(indices[split:]),
        num_workers=num_workers
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False, # shuffle,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader

def get_cifar100_dataloader(batch_size, num_workers, shuffle, resize=None):
    """
    :param batch_size: dataloader batchsize
    :param num_workers: dataloader num_works
    :param shuffle: flag of shuffle
    :param resize: resize
    :return: train dataloader & valid dataloader
    """

    # data transform
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    ratio = 0.9

    train_trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    valid_trans = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    if resize:
        train_trans.insert(1, transforms.Resize(size=resize))
        valid_trans.insert(0, transforms.Resize(size=resize))

    train_data = datasets.CIFAR100(
        root='./data/CIFAR100',
        train=True,
        transform=transforms.Compose(train_trans),
        download=False
    )

    valid_data = datasets.CIFAR100(
        root='./data/CIFAR100',
        train=True,
        transform=transforms.Compose(valid_trans),
        download=False
    )

    test_data = datasets.CIFAR10(
        root='./data/CIFAR100',
        train=False,
        transform=transforms.Compose(valid_trans),
        download=False
    )

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))
    split = int(np.floor(ratio * num_train))
    print('split ratio:{}, train nums:{}, valid nums:{}, test nums:{}'.format(ratio, split, num_train - split, len(test_data)))

    np.random.shuffle(indices)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=Data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=num_workers
    )

    valid_loader = Data.DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        sampler=Data.sampler.SubsetRandomSampler(indices[split:]),
        num_workers=num_workers
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False, # shuffle,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


def _data_transforms_spine3(cutout_size, autoaugment=False):
    pass

def build_search_spine3(args=None, root_path=None, cutout_size=None, batch_size=16, autoaugment=False, num_workers=0, retrain=False):

    # used for searching process, so valid_data "train=True"
    # train_transform, valid_transform = _data_transforms_spine3(cutout_size, autoaugment)

    if retrain:
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # resize
                transforms.RandomCrop(224, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )

        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # resize
                # transforms.RandomCrop(resize, padding=16),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # resize
                transforms.RandomCrop(128, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )

        valid_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # resize
                # transforms.RandomCrop(resize, padding=16),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )

    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    train_path = root_path + "train"
    valid_path = root_path + "valid"
    test_path = root_path + "test"


    train_data = dataset.ImageFolder(root=train_path, transform=train_transform)
    valid_data = dataset.ImageFolder(root=valid_path, transform=valid_transform)
    test_data = dataset.ImageFolder(root=test_path, transform=valid_transform)


    num_train = len(train_data)
    num_valid = len(valid_data)
    num_test = len(test_data)

    # {'normal':0, 'serious':1, 'slight':2}
    spine_list = train_data.class_to_idx

    # batch size
    # batch_size = 32

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, # 32
        shuffle=True,
        num_workers=num_workers
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_queue, valid_queue, test_queue

def build_train_cifar10(args, cutout_size=None, autoaugment=False):
    # used for training process, so valid_data "train=False"

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)

    train_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    return train_queue, valid_queue

def build_train_cifar100(args, cutout_size=None, autoaugment=False):

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)
    train_data = dataset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    return train_queue, valid_queue


# ==================== Optimizer_Loss ====================
def build_search_Optimizer_Loss(model, args, epoch=-1):
    model.cuda() # gpu
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    # SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.search_lr_max,
        momentum=args.search_momentum,
        weight_decay=args.search_l2_reg,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.search_epochs, args.search_lr_min, epoch)

    return  train_criterion, eval_criterion, optimizer, scheduler

def build_train_Optimizer_Loss(model, args, epoch=-1):
    model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=args.momentum,
        weight_decay=args.l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)

    return train_criterion, eval_criterion, optimizer, scheduler



