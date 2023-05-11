import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_image_size(dataset_name):
    if dataset_name == "CIFAR10":
        image_size = 32
    elif dataset_name == "CIFAR100":
        image_size = 32
    else:
        raise NotImplementedError("Dataset {} not supported".format(dataset_name))
    return image_size


def get_class_nums(args):
    if args.dataset_name == "CIFAR10":
        class_nums = 10
    elif args.dataset_name == "CIFAR100":
        class_nums = 100
    else:
        raise NotImplementedError("Dataset {} not supported".format(args.dataset_name))
    return class_nums
