
from PIL import Image
import numpy as np

# DATA_PREFIX = "/home/vradmin/Desktop/paper_work/EM/data/cones/"
DATA_PREFIX = "/home/vradmin/Desktop/paper_work/EM/data/Aloe/"
FACTOR = 9

## 读取图片和disparity map
# true_disparity_image = (
#     np.array(Image.open(DATA_PREFIX + "disp2_small.png")) / FACTOR).astype(int)
# image1 = np.array(Image.open(
#     DATA_PREFIX + "im2_small.png").convert("L"), dtype='int64')
# image2 = np.array(Image.open(
#     DATA_PREFIX + "im6_small.png").convert("L"), dtype='int64')

true_disparity_image = np.array(Image.open(DATA_PREFIX + "disp1.png")).astype(int)
image1 = np.array(Image.open(DATA_PREFIX + "view1_small.png").convert("L"), dtype='int64')
image2 = np.array(Image.open(DATA_PREFIX + "view5_small.png").convert("L"), dtype='int64')

#############################
""" basic params """

NUM_INPUT = 2
HEIGHT = image1.shape[0]
WIDTH = image1.shape[1]
NUM_COLOR = 2  # 灰度图

# 32 种 color bins(0,1,2,3,4,5,6,7 是第一个bin), 灰度图的 scale 0 - 255
NUM_COLOR_BINS = 256
num_color_in_bin = 256 / NUM_COLOR_BINS
C = 10 ** (-10)  # in the psi function(potential function)
sigma_d = 0.03
sigma_v = 0.3

###############################
""" visible variables """

NUM_VISIBLE_CONF = 1  # S
NUM_DEPTH_LEVEL = 24  # R depth level
num_visible_state = NUM_VISIBLE_CONF * \
    NUM_DEPTH_LEVEL  # M = R * S 从 0 开始 m = r * S + s

## visibility configuration matrix
# visibility_conf_mat = np.array([[1, 1], [1, 1]])
visibility_conf_mat = np.array([[1, 1]])

## depth value for each level
depth_vec = [i for i in range(NUM_DEPTH_LEVEL)]

###############################
""" input image """

I = [image1, image2]


###############################
""" Theta"""

## ideal image
ideal_image = I[0].copy()  # 初始值设为 I[0]


## histgram for each image
## 0(0,1,2,3,4,5,6,7), 1(8,9,10,11,12,13,14,15), ... 31 ( 32 bins )
## 初始分布暂时选取一个均匀分布
hist_mat = np.ones([NUM_INPUT, NUM_COLOR_BINS]) / NUM_COLOR_BINS


## 灰度图相当于一个一维分布, 不能随意初始化，否则会导致正太分布的概率过于小
## 没有办法的办法，即便我在别的模块中饮用 config, 其实获得的也是copy, 导致 covariance 无法更新
covariance = [10]
###############################


### b_mat, visible_state, disparity_image, visible_image

b_mat = np.zeros([HEIGHT * WIDTH, num_visible_state])
visible_state = np.zeros([HEIGHT, WIDTH])
disparity_image = np.zeros([HEIGHT, WIDTH])
# disparity_image = true_disparity_image.copy()
visible_image = np.zeros([HEIGHT, WIDTH])




