import argparse
import logging
import math
import os
import random
import shutil
from threading import local
import time
from collections import OrderedDict
import numpy as np
import sys

import paddle
import paddle.nn.functional as F

from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler, DistributedBatchSampler

from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

best_acc = 0
global local_master, logger


def get_logger(args, name=__name__, verbosity=2):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[verbosity] if args.local_rank in [-1, 0] else logging.INFO,
                        filename=f'{args.log_out}/train@{args.num_labeled}.log',
                        filemode='a')

    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    return logger


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pdparams'):
    filepath = os.path.join(checkpoint, filename)
    paddle.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pdparams'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose([1, 0, 2, 3, 4]).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose([1, 0, 2]).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='Paddle FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--val-iter', default=64, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--log-out', default='logs',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model-best', default='', type=str,
                        help='path to best checkpoint to test (default: none)')
    parser.add_argument('--data-file', default='/paddle/data/cifar-10-python.tar.gz', type=str,
                        help='path to cifar10 dataset')
    parser.add_argument('--seed', default=5, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--device', default=paddle.get_device(),
                        help="don't use progress bar")

    args = parser.parse_args()

    global local_master, logger
    local_master = (args.local_rank == -1 or dist.get_rank() == 0)

    if local_master:
        os.makedirs(args.log_out, exist_ok=True)

    logger = get_logger(args) if local_master else None

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            (sum(p.numel() for p in model.parameters()) / 1e6).numpy()[0])) if local_master else None
        return model

    if args.seed is not None:
        set_seed(args)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, args.data_file)
    test_sampler = SequenceSampler(test_dataset)
    test_batch_sampler = BatchSampler(sampler=test_sampler,
                                      batch_size=args.batch_size * args.mu,
                                      drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=args.num_workers)

    model = create_model(args)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model)

    if args.model_best:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.model_best), "Error: no checkpoint directory found!"
        checkpoint = paddle.load(args.model_best)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.set_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.set_state_dict(checkpoint['ema_state_dict'])

    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model

    test_model.eval()

    if local_master:
        predict(args, test_loader, test_model)


def predict(args, test_loader, model):
    global local_master, logger

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with paddle.no_grad():
        batch_idx, (inputs, targets) = next(enumerate(test_loader))
        print(inputs.shape)
        inputs = inputs[0].unsqueeze(0)
        data_time.update(time.time() - end)
        model.eval()

        outputs = model(inputs)
        # outputs = F.softmax(outputs)
        class_id = paddle.argmax(outputs, axis=1)
        print(class_id)
        print(outputs)
        prob = outputs[class_id.item()]
        print(class_id)
        print(prob)


if __name__ == '__main__':
    main()
