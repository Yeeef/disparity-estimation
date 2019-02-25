import argparse
import os
import numpy as np

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

import tensorflow as tf

def resize_image(image, out_height, out_width, data_format='channels_first'):
    assert data_format in ['channels_first', 'channels_last']
    if data_format == 'channels_last':
        return tf.image.resize_images(image, [out_height, out_width])
    else:
        # batch_size, channel, height, width -> batch, h, w, ch
        tmp_img = tf.transpose(image, [0, 2, 3, 1])
        tmp_img = tf.image.resize_images(
            tmp_img, [out_height, out_width])
        # b, h, w, c -> b, c, h, w
        return tf.transpose(tmp_img, [0, 3, 1, 2])

# without bottleneck block


def residual(name, l, increase_dim=False, first=False):
    shape = l.get_shape().as_list()
    # channel first NCHW
    in_channel = shape[1]

    # 是否增加维度(channel 的个数)
    # it is for the strided conv layer
    if increase_dim:
        out_channel = in_channel * 2
        stride1 = 2
    else:
        out_channel = in_channel
        stride1 = 1

    with tf.variable_scope(name):
        b1 = l if first else BNReLU(l)
        c1 = Conv2D('conv1', b1, out_channel,
                    strides=stride1, activation=BNReLU)
        # the default value of strides is 1
        # the kernel size is defined in the scope of `argscope`
        # the default activation is `identity mapping`
        c2 = Conv2D('conv2', c1, out_channel)
        # increase the dimension of the channel
        # at the same time, the size of the feature map is halved
        if increase_dim:
            # half the size of the feature map
            l = AvgPooling('pool', l, 2)
            # increase the dimension of channel with 0 filled
            # we can also use a 1 * 1 conv layer to do this
            l = tf.pad(
                l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

        l = c2 + l
        return l


def self_image_gradients(image, data_format='channels_last'):
    assert data_format in ['channels_last', 'channels_first']
    if data_format == 'channels_last':
        return tf.image.image_gradients(image)
    else:
        # image: b, c, h, w
        tmp_img = tf.transpose(image, [0, 2, 3, 1])
        tmp_dy, tmp_dx = tf.image.image_gradients(tmp_img)
        return (tf.transpose(tmp_dy, [0, 3, 1, 2]), tf.transpose(tmp_dx, [0, 3, 1, 2]))

# depth map and output are all 'b*H*W' (b denotes 'batch')
def get_loss(depth_map, output, alpha, name="loss"):
    D, G_x, G_y = get_D_and_G(depth_map, output)
    return tf.add(get_l1(D, G_x, G_y, 'l1_loss'), alpha * get_l2(D, G_x, G_y, 'l2_loss'), name=name)


def get_l1(D, G_x, G_y, name='l1_loss'):

    return tf.reduce_mean(tf.add_n([tf.abs(D), tf.abs(G_x), tf.abs(G_y)]), name=name)


def get_l2(D, G_x, G_y, name='l2_loss'):

    return tf.reduce_mean(tf.add_n([tf.square(D), tf.square(G_x), tf.square(G_y)]), name=name)

# return D, G_x, G_y
def get_D_and_G(depth_map, output):
    # batch_size, height, width = depth_map.get_shape().as_list()[:3]

    D = pixel_loss(depth_map, output, name='D')
    # b * H * W * 1
    dy_depth_map, dx_depth_map = self_image_gradients(
        tf.expand_dims(depth_map, 3),
        data_format='channels_last')
    
    dy_depth_map = tf.squeeze(dy_depth_map, axis=3)
    dx_depth_map = tf.squeeze(dx_depth_map, axis=3)

    # b * H * W * 1
    dy_output, dx_output = self_image_gradients(
        tf.expand_dims(output, 3),
        data_format='channels_last'
    )
    dy_output = tf.squeeze(dy_output, axis=3)
    dx_output = tf.squeeze(dx_output, axis=3)
    G_y = pixel_loss(dy_depth_map, dy_output, name='G_y')
    G_x = pixel_loss(dx_depth_map, dx_output, name='G_x')

    return (D, G_x, G_y)

def get_rank_loss(depth_map, output):
    pass

# depth map and output are all 'b*H*W' (b denotes 'batch')
def pixel_loss(depth_map, output, name):
    _, height, width = depth_map.get_shape().as_list()
    # D(I,i) = log(Z_i) - log(I_i^d) - (1 / N) * \Sigma(log(Z_j) - log(I_j^d))
    # b*H*W
    log_depth_map = tf.log(depth_map)
    # b*H*W
    log_output = tf.log(output)
    # b * H * W
    sub_log = tf.subtract(log_output, log_depth_map)

    # b * 1 * 1
    # every feature map of a batch owns a mean value
    tmp1 = tf.reduce_mean(sub_log, [1, 2], keepdims=True)
    # b * H * W
    tmp1 = tf.tile(tmp1, [1, height, width])

    # b * H * W
    return tf.subtract(sub_log, tmp1, name=name)
