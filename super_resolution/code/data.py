# -*- coding: utf-8 -*-
# File: data.py
# Author: Yeeef

import os
import numpy as np
import pickle

from tensorpack import *

__all__ = ['NYUBase', 'NYUV1', 'NYUV2']

def read_NYU(filenames):
    ret = []
    for file in filenames:
        with open(file, 'rb') as f:
            # rgb concat with depth
            # channels_first
            crgb = pickle.load(f)
        ret.append(crgb)
    return ret
        

class NYUBase(RNGDataFlow):
    """
    produces [image, depth] in NYU dataset,
    image is 3 * 640 * 480 in range [0, 255]
    depth is 640 * 480 in range [0, 255]
    """

    def __init__(self, dir, train_or_test, shuffle=None):
        """
        Args:
            * train_or_test (str): 'train' or 'test'
            * shuffle (bool): defaults to True for training set
            * dir (str): path to the dataset(pickle) directory
        """
        assert train_or_test in ['train', 'test']
        train_files = [os.path.join(dir, f'{i}.pickle') for i in range(1200)]
        test_files = [os.path.join(dir, f'{i}.pickle') for i in range(1200, 1449)]
        if train_or_test == 'train':
            self.fs = train_files
        else:
            self.fs = test_files
        
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError(f'Failed to find file {f}')
        
        self.train_or_test = train_or_test
        self.data = read_NYU(self.fs)
        self.dir = dir
        
        if shuffle is None:
            shuffile = (train_or_test == 'train')
        self.shuffle = shuffle



    def __len__(self):
        return 1200 if self.train_or_test == 'train' else 249

    def __iter__(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.data[k]
    
    def get_per_pixel_mean(self, names=('train', 'test')):
        # b, c, h, w
        all_imgs = np.array([x[:3] for x in self.data], dtype=np.float32)
        all_depths = np.array([x[3] for x in self.data], dtype=np.float32)
        # 3, h, w
        img_mean = np.mean(all_imgs, axis=0)
        # 1, h, w
        depth_mean = np.mean(all_depths, axis=0)

        return (img_mean, depth_mean)

    

class NYUV1(NYUBase):
    pass

class NYUV2(NYUBase):
    pass

if __name__ == "__main__":
    ds = NYUBase('/Users/yee/Desktop/NYUv2', 'test')
    print(len(ds))
    img_mean, depth_mean = ds.get_per_pixel_mean()
    print(img_mean)
    print(depth_mean)