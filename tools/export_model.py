# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from paddle.static import InputSpec

import paddle
import paddle.nn.functional as F

from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler, DistributedBatchSampler

from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from dataset.cifar import DATASET_GETTERS


def get_args(add_help=True):
    """get_args

    Parse all args using argparse lib

    Args:
        add_help: Whether to add -h option on args

    Returns:
        An object which contains many parameters used for inference.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument(
        '--save-inference-dir', default='.', help='path where to save')

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
    parser.add_argument('--img-size', default=32, help='image size to export')
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
                        help="device")

    args = parser.parse_args()
    return args


def create_model(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

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
    return model


def export(args):
    # build model
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        paddle.seed(args.seed)
    model = create_model(args)
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model)

    if args.model_best:
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

    # decorate model with jit.save
    model = paddle.jit.to_static(
        test_model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)
