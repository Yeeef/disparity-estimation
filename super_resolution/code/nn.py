#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: nn.py

"""
network architecture settings
backbone: resnet
"""

from self_utils import *
from data import *

import argparse
import os
import numpy as np

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

from resnet_model import *


import tensorflow as tf


BATCH_SIZE = 128
STEPS_PER_EPOCH = 1449 // BATCH_SIZE
HEIGHT = 480
WIDTH = 640


class Model(ModelDesc):

    def __init__(self):
        super(Model, self).__init__()
        

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 4, HEIGHT, WIDTH], 'input_img_depth')]

    def build_graph(self, img_depth):
        # b, c, h, w
        image = img_depth[:, :3]
        # b, h, w
        original_depth_map = img_depth[:, 3]

        # b, h, w, 1
        depth_map = resize_image(tf.expand_dims(
            original_depth_map, 3), HEIGHT//2, WIDTH//2, 'channels_last')
        # b, 1, h, w
        depth_map = tf.transpose(depth_map, [0, 3, 1, 2])

        # assert tf.test.is_gpu_available()

        def _resnet_backbone(image, num_blocks, group_func, block_func):
            with argscope(Conv2D, use_bias=False,
                          kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
                # Note that this pads the image by [2, 3] instead of [3, 2].
                # Similar things happen in later stride=2 layers as well.
                l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
                l = MaxPooling('pool0', l, pool_size=3,
                               strides=2, padding='SAME')
                l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
                l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
                l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
                l = group_func('group3', l, block_func, 512, num_blocks[3], 2)

            return l

        def resnet50(image):
            return _resnet_backbone(image, [3, 4, 6, 3], resnet_group, resnet_bottleneck)

        with argscope([Conv2D, MaxPooling, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            """
            Down-sampling part:
                * Resnet50 as backbone
                * use pre-trained ResNet50 in <http://models.tensorpack.com/>
            """
            # 2048*H/32*W/32
            l = resnet50(image)

            """
            Up-sampling part
                * up-sampling block adopt the spirit of residual block,
                  where we create a direct path for information passing.
                * we use the method of preactivation
            """
            # 1024*H/32*W/32
            l = Conv2D('conv1', l, filters=1024, activation=BNReLU)

            # first up-sampling block
            with tf.variable_scope("up1.0"):

                # @WARN: resize_image_with_pad only accept the image with the format of channels last
                # here I use up-sampling rather than unpooling
                # 1024*H/16*W/16
                l = resize_image(l, HEIGHT//16, WIDTH//16)

                # 512*H/16*W/16
                shortcut = Conv2D('conv0', l, filters=512, kernel_size=1)

                c1 = Conv2D('conv1', l, filters=512, activation=BNReLU)
                c2 = Conv2D('conv2', c1, filters=512, activation=BNReLU)
                c3 = Conv2D('conv3', c2, filters=512)

                # 512*H/16*W/16
                # elementwise SUM layer
                l = shortcut + c3

            with tf.variable_scope("up2.0"):
                # pre-act
                l = BNReLU(l)

                # 512*H/8*W/8
                l = resize_image(l, HEIGHT//8, WIDTH//8)
                # the channels does not decrease, just simply pass the information
                shortcut = l

                side_input = resize_image(depth_map, HEIGHT//8, WIDTH//8)
                side_input = Conv2D('stride_conv0', side_input,
                                    64, kernel_size=1, activation=BNReLU)

                concat_ret = tf.concat(values=[l, side_input], axis=1)

                c1 = Conv2D('conv1', concat_ret, 512, activation=BNReLU)
                c2 = Conv2D('conv2', c1, 512, activation=BNReLU)
                c3 = Conv2D('conv3', c2, 512)

                # 512*H/8*W/8
                l = shortcut + c3

            with tf.variable_scope('up3.0'):
                # pre-act
                l = BNReLU(l)
                # 512*H/8*W/8
                side_output0 = Conv2D('multi_conv0', l, 1, kernel_size=1)

                # 512*H/4*W/4
                l = resize_image(l, HEIGHT//4, WIDTH//4)
                shortcut = Conv2D('conv0', l, 256, kernel_size=1)

                side_input = resize_image(depth_map, HEIGHT//4, WIDTH//4)
                side_input = Conv2D('stride_conv0', side_input,
                                    64, kernel_size=1, activation=BNReLU)
                concat_ret = tf.concat(values=[l, side_input], axis=1)

                # 256*H/4*W/4
                c1 = Conv2D('conv1', concat_ret, 256, activation=BNReLU)
                c2 = Conv2D('conv2', c1, 256, activation=BNReLU)
                c3 = Conv2D('conv3', c2, 256)

                # 256*H/4*W/4
                l = shortcut + c3

            with tf.variable_scope('up4.0'):
                # pre-act
                l = BNReLU(l)

                # b*1*H/4*W/4
                side_output1 = Conv2D('multi_conv1', l, 1, kernel_size=1)

                # 256*H/2*W/2
                l = resize_image(l, HEIGHT//2, WIDTH//2)
                shortcut = Conv2D('conv0', l, 128, kernel_size=1)

                # @WARN: actually, the default is H/2*W/2 for depth map
                side_input = resize_image(depth_map, HEIGHT//2, WIDTH//2)
                side_input = Conv2D('stride_conv0', side_input,
                                    64, kernel_size=1, activation=BNReLU)
                concat_ret = tf.concat(values=[l, side_input], axis=1)

                # 128*H/2*W/2
                c1 = Conv2D('conv1', concat_ret, 128, activation=BNReLU)
                c2 = Conv2D('conv2', c1, 128, activation=BNReLU)
                c3 = Conv2D('conv3', c2, 128)

                l = shortcut + c3

            with tf.variable_scope('output'):
                # pre-act
                l = BNReLU(l)
                # 128*H/2*W/2
                side_output2 = Conv2D('multi_conv1', l, 1, kernel_size=1)

                # 128*H*W
                l = resize_image(l, HEIGHT, WIDTH)

                # 64*H*W
                c1 = Conv2D('conv1', l, 64, activation=BNReLU)
                c2 = Conv2D('conv2', c1, 64, activation=BNReLU)

                # b*1*H*W
                output = Conv2D('output_conv', c2, 1,
                                kernel_size=1, activation=BNReLU)

        # channels_first
        batch_size, _, height, width = output.get_shape().as_list()
        # H/8*W/8
        loss0 = get_loss(
            tf.squeeze(resize_image(tf.expand_dims(original_depth_map, 1), HEIGHT//8, WIDTH//8, 'channels_first'), axis=1),
            tf.squeeze(side_output0, axis=1),
            # tf.reshape(side_output0, [batch_size, height, width]),
            name='loss0',
            alpha=0.5
        )

        loss1 = get_loss(
            tf.squeeze(resize_image(tf.expand_dims(original_depth_map, 1), HEIGHT//4, WIDTH//4, 'channels_first'), axis=1),
            tf.squeeze(side_output1, axis=1),
            name='loss1',
            alpha=0.5
        )

        loss2 = get_loss(
            tf.squeeze(resize_image(tf.expand_dims(original_depth_map, 1), HEIGHT//2, WIDTH//2, 'channels_first'), axis=1),
            tf.squeeze(side_output2, axis=1),
            name='loss2',
            alpha=0.5
        )

        loss_output = get_loss(
            original_depth_map,
            tf.squeeze(output, axis=1),
            name='loss_output',
            alpha=0.5
        )

        add_moving_summary(loss0, loss1, loss2, loss_output)
        add_param_summary(('.*/W', ['histogram']))

        return tf.add_n([loss0, loss1, loss2, loss_output], name='loss')

    def optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=0.01, trainable=False)
        return tf.train.MomentumOptimizer(lr, 0.9)

    def _preprocess_image():
        pass


def get_data(input_dir, train_or_test):
    assert train_or_test in ['train', 'test']
    is_train = (train_or_test == 'train')
    ds = NYUBase(input_dir, train_or_test)
    # img_mean, depth_mean = ds.get_per_pixel_mean()
    # concat_mean = np.concatenate([img_mean, depth_mean], axis=0)

    if is_train:
        augmentors = [
            # imgaug.RandomCrop(128),
            imgaug.MapImage(
                lambda x: x - concat_mean),
            imgaug.Rotation(5),
            imgaug.Flip(horiz=True)
        ]
    else:
        augmentors = [
            # imgaug.MapImage(
            #     lambda x: x - concat_mean)
        ]
    # ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not is_train)
    # if is_train:
    #     ds = PrefetchData(ds, 3, 2)
    return ds

def get_train_conf(dataset, is_gpu, session_init=None):
    callbacks = [
        ModelSaver(),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)]),
    ]
    if is_gpu:
        callbacks.append(GPUUtilizationTracker())
    conf = TrainConfig(
        model=Model(),
        # data=QueueInput(dataset),
        data=FeedInput(dataset),
        callbacks=callbacks,
        max_epoch=400,
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=session_init
    )
    return conf


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to the NYU data', default='Users/yee/Desktop/NYUv2')
    parser.add_argument('--cpu', help='just for debug', action='store_true')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--resnet50', help='path of the pre-trained resnet50 in .npz form')
    parser.add_argument('--load', help='path of the model to load')
    parser.add_argument('--apply', help='not train, please be sure to supply --load argument too')
    args = parser.parse_args()

    if args.cpu:
        is_gpu = False
        logger.auto_set_dir(action='d')
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        logger.auto_set_dir()
        is_gpu = True

    if args.apply:
        pass
    else:
        session_init = None
        if args.resnet50:
            session_init = get_model_loader(args.resnet50)
        if args.load:
            session_init = get_model_loader(args.load)
        
        input_dir = args.data
        ds = get_data(input_dir, 'train')
        conf = get_train_conf(ds, is_gpu, session_init)

        if args.cpu:
            trainer = SimpleTrainer()
        else:
            trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
        launch_train_with_config(conf, trainer)
