import utils
import numpy as np

#############################
""" basic params """

NUM_INPUT = 3
HEIGHT = 480
WIDTH = 640
NUM_COLOR = 2  # 灰度图
# 32 种 color bins(0,1,2,3,4,5,6,7 是第一个bin), 灰度图的 scale 0 - 255
NUM_COLOR_BINS = 32
num_color_in_bin = 256 / NUM_COLOR_BINS
C = 10 ^ (-4)  # in the psi function(potential function)

###############################
""" visible variables """

NUM_VISIBLE_CONF = 4  # S
NUM_DEPTH_LEVEL = 20  # R depth level
num_visible_state = NUM_VISIBLE_CONF * NUM_DEPTH_LEVEL  # M = R * S 从 1 开始

## visibility configuration matrix
visibility_conf_mat = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])

## depth value for each level
depth_vec = [i + 1 for i in range(NUM_DEPTH_LEVEL)]

visible_state = np.random.randint(low=1, high=num_visible_state, size=[HEIGHT, WIDTH])

## E step 中的 b 矩阵，用于量化隐变量分布
## nrows=height*width(相当于一维化的坐标), ncols=M
b_mat = np.random.rand(HEIGHT*WIDTH, num_visible_state)




###############################
""" input image and ideal image """
image_list, ideal_K_mat, ideal_R, ideal_T = utils.construct_basic_param(3)

## ideal image
ideal_image = np.random.randint(low=0, high=256, size=[HEIGHT, WIDTH])
ideal_calibration = ideal_K_mat, ideal_R, ideal_T

I = [image.I for image in image_list]
calibration_list = [[image.K_mat, image.rotation_mat,
                     image.translation_vec] for image in image_list]
K_mat = [image.K_mat for image in image_list]
R_mat = [image.rotation_mat for image in image_list]
T_vec = [image.translation_vec for image in image_list]

## histgram for each image
## 0(0,1,2,3,4,5,6,7), 1(8,9,10,11,12,13,14,15), ... 31 ( 32 bins )
hist_mat = np.random.randint(low=0, high=256, size=[NUM_INPUT, NUM_COLOR_BINS])



## 灰度图相当于一个一维分布
covariance = np.random.rand()


