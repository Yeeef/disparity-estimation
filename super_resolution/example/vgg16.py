#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vgg16.py

# import argparse
# import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu
from tensorflow import keras
from imagenet_utils import (
    ImageNetModel, get_imagenet_dataflow, fbresnet_augmentor)
