# -*- coding: utf-8 -*-
# File: data.py
# Author: Yeeef

import os
import numpy as np
import pickle

from tensorpack import (RNGDataFlow)

__all__ = ['NYUBase', 'NYUV1', 'NYUV2']


def read_NYU(filenames, data_format):
    ret = []
    assert data_format in ['channels_first', 'channels_last'], data_format
    for file in filenames:
        with open(file, 'rb') as f:
            # rgb concat with depth
            # channels_first 4,h,w
            crgb = pickle.load(f)
            # channels_last h,w,4
            if data_format == 'channels_last':
                crgb = np.transpose(crgb, [1, 2, 0])

        ret.append(crgb)
    return ret


class NYUBase(RNGDataFlow):
    """
    produces [image concat depth] in NYU dataset,
    image is 3 * 480 * 640 in range [0, 255]
    depth is 480 * 640 in range [0, 255]
    image concat depth is 4 * 480 * 640
    """

    def __init__(self, dir, train_or_test, data_format, shuffle=None):
        """
        Args:
            * train_or_test (str): 'train' or 'test'
            * shuffle (bool): defaults to True for training set
            * dir (str): path to the dataset(pickle) directory
        """
        assert train_or_test in ['train', 'test']
        train_files = [os.path.join(dir, f'{i}.pickle') for i in range(1200)]
        test_files = [os.path.join(dir, f'{i}.pickle')
                      for i in range(1200, 1449)]
        if train_or_test == 'train':
            self.fs = train_files
        else:
            self.fs = test_files

        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError(f'Failed to find file {f}')

        self.train_or_test = train_or_test
        self.data_format = data_format
        self.data = read_NYU(self.fs, self.data_format)
        self.dir = dir

        if shuffle is None:
            shuffle = (train_or_test == 'train')
        self.shuffle = shuffle

    def __len__(self):
        return 1200 if self.train_or_test == 'train' else 249

    def __iter__(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k]]

    def get_per_pixel_mean(self, names=('train', 'test')):
        assert self.data_format in ['channels_first', 'channels_last'], self.data_format
        if self.data_format == 'channels_first':
            # 1200, 3, h, w
            all_imgs = np.array([x[:3] for x in self.data], dtype=np.float32)
            # 1200, h, w
            all_depths = np.array([x[3] for x in self.data], dtype=np.float32)
            # 3, h, w
            img_mean = np.mean(all_imgs, axis=0)
            # print(img_mean.shape)
            # 1, h, w
            depth_mean = np.expand_dims(np.mean(all_depths, axis=0), 0)
            # print(depth_mean.shape)
            assert img_mean.shape == (3, 480, 640)
            assert depth_mean.shape == (1, 480, 640)
            return (img_mean, depth_mean)
        else:
            all_imgs = np.array(
                [x[:,:,:3] for x in self.data], dtype=np.float32
            )
            all_depths = np.array(
                [x[:,:,3] for x in self.data], dtype=np.float32
            )
            # h, w, 3
            img_mean = np.mean(all_imgs, axis=0)
            # h, w, 1
            depth_mean = np.expand_dims(np.mean(all_depths, axis=0), 2)
            assert img_mean.shape == (480, 640, 3)
            assert depth_mean.shape == (480, 640, 1)

            return (img_mean, depth_mean)


class NYUV1(NYUBase):
    pass


class NYUV2(NYUBase):
    pass


if __name__ == "__main__":
    ds = NYUBase('/Users/yee/Desktop/NYUv2', 'test', "channels_last")
    print(len(ds))
    img_mean, depth_mean = ds.get_per_pixel_mean()
    print(img_mean.shape)
    print(depth_mean.shape)
    print('=' * 20)
    i = 0
    for k in ds:
        print(np.array(k).shape)
        print(k)
        print(type(k))
        i = i + 1
        if i == 1:
            break
