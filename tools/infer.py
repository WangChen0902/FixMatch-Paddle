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

import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

import argparse
import logging
import math
import random
import shutil
from threading import local
import time
from collections import OrderedDict
import sys

import paddle.nn.functional as F

from paddle.io import DataLoader, RandomSampler, SequenceSampler, BatchSampler, DistributedBatchSampler

from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from reprod_log import ReprodLogger

class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess

    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor

        initialize the inference engine

        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, args):
        """preprocess

        Preprocess to the input.

        Args:
            img_path: Image path.

        Returns: Input data after preprocess.
        """
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)

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
        batch_idx, (inputs, targets) = next(enumerate(test_loader))
        print(inputs.shape)
        inputs = inputs[0].numpy()
        inputs = np.expand_dims(inputs, axis=0)
        return inputs

    def postprocess(self, x):
        """postprocess

        Postprocess to the inference engine output.

        Args:
            x: Inference engine output.

        Returns: Output data after argmax.
        """
        x = x.flatten()
        class_id = x.argmax()
        prob = x[class_id]
        return class_id, prob

    def run(self, x):
        """run

        Inference process using inference engine.

        Args:
            x: Input data after preprocess.

        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default='./infer', help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--max-batch-size", default=16, type=int, help="max_batch_size")

    parser.add_argument(
        "--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=32, type=int, help="crop_szie")
    parser.add_argument("--img-path", default="/paddle/data/cifar-10-python.tar.gz")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")

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


def infer_main(args):
    """infer_main

    Main inference function.

    Args:
        args: Parameters generated using argparser.

    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    class_id, prob = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")
    return class_id, prob


if __name__ == "__main__":
    args = get_args()
    class_id, prob = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("class_id", np.array([class_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.save("output_inference_engine.npy")
