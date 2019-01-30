#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: nn.py

"""
network architecture settings
backbone: resnet
"""

import argparse
import os
import numpy as np

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu

import tensorflow as tf


BATCH_SIZE = 128


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input_image'),
                tf.placeholder(tf.float32, [None, 16, 16, 1], 'input_depthmap')]

    def build_graph(self, image, depth_map):
        image = self._preprocess_image(image)
        depth_map = self._preprocess_image(depth_map)
        assert tf.test.is_gpu_available()

        # channel first, why?

        # residual block

        def residual(name, l, increase_dim=False, first=False):
            pass

        with argscope(Conv2D, use_bias=False, kernel_size=3,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):

            # H*W
            l = Conv2D('conv0', image, filters=64, activation=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual(f"res1.{k}", l)
            
            """
            down-sampling
            """
            # 128*H/2*W/2           
            l = residual("res2.0", l, increase_dim=True)
            for k in range(1, self.n):
                l = residual(f"res2.{k}", l)
            
            # 256*H/4*W/4            
            l = residual("res3.0", l, increase_dim=True)
            for k in range(1, self.n):
                l = residual(f"res3.{k}", l)

            # 512*H/8*W/8
            l = residual("res4.0", l, increase_dim=True)
            for k in range(1, self.n):
                l = residual(f"res4.{k}", l)

            # 1024*H/16*W/16
            l = residual("res5.0", l, increase_dim=True)
            for k in range(1, self.n):
                l = residual(f"res5.{k}", l)
            
            # 2048*H/32*W/32
            l = residual("res6.0", l, increase_dim=True)
            for k in range(1, self.n):
                l = residual(f"res6.{k}", l)
            
            l = BNReLU('down_bnlast', l)
            l = Conv2D('conv1', l, filters=1024, activation=BNReLU)

            """
            up-sampling
            """
            def upsample(x, factor=2):
                _, h, w, _ = x.get_shape().as_list()
                x = tf.image.resize_nearest_neighbor(x, [factor * h, factor * w], align_corners=True)
            return x





            
            


            

        output_depthmap = l




        


        

    def _preprocess_image():
        pass
        
    

if __name__ == "__main__":
    pass

