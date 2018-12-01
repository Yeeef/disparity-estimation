from PIL import Image
import numpy as np
import os
import re

CONFIG_PATH = "/Users/yee/Desktop/paper_work/EM/result/Aloe/test4/CONFIG"

## 读取图片和disparity map
# true_disparity_image = (
#     np.array(Image.open(os.path.join(DATA_PREFIX, "disp2.png"))) / FACTOR).astype(int)
# image1 = np.array(Image.open(
#     os.path.join(DATA_PREFIX, "im2.png")).convert("L"), dtype='int64')
# image2 = np.array(Image.open(
#     os.path.join(DATA_PREFIX, "im6.png")).convert("L"), dtype='int64')

data_prefix_reg = re.compile(r"(DATA_PREFIX)[\s=]+([\w/\"]+)")
mode_reg = re.compile(r"(MODE)[\s=]+([a-zA-Z]+)")
n_color_bin_reg = re.compile(r"(NUM_COLOR_BINS)[\s=]+([\w]+)")
c_reg = re.compile(r"(C)[\s=]+([\d]+)[\^\s]+([-\d]+)")
sigma_d_reg = re.compile(r"(sigma_d)[\s=]+([.\d]+)")
sigma_v_reg = re.compile(r"(sigma_v)[\s=]+([.\d]+)")
occlusion_reg = re.compile(r"(OCCLUSION)[\s=]+([\w]+)")
covariance_reg = re.compile(r"(COVARIANCE)[\s=]+([\w]+)")

with open(CONFIG_PATH, "r") as infile:
    file_content = infile.read()
    data_prefix_search = re.search(data_prefix_reg, file_content)
    mode_search = re.search(mode_reg, file_content)
    n_color_bin_search = re.search(n_color_bin_reg, file_content)
    c_search = re.search(c_reg, file_content)
    sigma_d_search = re.search(sigma_d_reg, file_content)
    sigma_v_search = re.search(sigma_v_reg, file_content)
    occlusion_search = re.search(occlusion_reg, file_content)
    covariance_search = re.search(covariance_reg, file_content)


DATA_PREFIX = data_prefix_search.group(2)
print(DATA_PREFIX)
MODE = mode_search.group(2)
print(MODE)
NUM_COLOR_BINS = int(n_color_bin_search.group(2).strip())
print(NUM_COLOR_BINS)
C = int(c_search.group(2)) ** int(c_search.group(3))
print(C)
sigma_d = float(sigma_d_search.group(2))
print(sigma_d)
sigma_v = float(sigma_v_search.group(2))
print(sigma_v)
OCCLUSION = int(occlusion_search.group(2))
print(OCCLUSION)
COVARIANCE = int(covariance_search.group(2))
print(COVARIANCE)

if MODE.strip().lower() == "small":
    FACTOR = 9
    NUM_DEPTH_LEVEL = 24
    true_disparity_image = np.array(Image.open(
        os.path.join(DATA_PREFIX, "disp1.png"))).astype(int)
    image1 = np.array(Image.open(
        os.path.join(DATA_PREFIX, "view1_small.png")).convert("L"), dtype='int64')
    image2 = np.array(Image.open(
        os.path.join(DATA_PREFIX, "view5_small.png")).convert("L"), dtype='int64')
    covariance = [COVARIANCE]

elif MODE.strip().lower() == "large":
    FACTOR = 3
    NUM_DEPTH_LEVEL = 72
    true_disparity_image = np.array(Image.open(
        os.path.join(DATA_PREFIX, "disp1.png"))).astype(int)
    image1 = np.array(Image.open(
        os.path.join(DATA_PREFIX, "view1.png")).convert("L"), dtype='int64')
    image2 = np.array(Image.open(
        os.path.join(DATA_PREFIX, "view5.png")).convert("L"), dtype='int64')
    covariance = [COVARIANCE]
else:
    print(f"Wrong mode: {MODE}")

NUM_INPUT = 2
HEIGHT = image1.shape[0]
WIDTH = image1.shape[1]
NUM_COLOR = 2  # 灰度图

# 32 种 color bins(0,1,2,3,4,5,6,7 是第一个bin), 灰度图的 scale 0 - 255
num_color_in_bin = 256 / NUM_COLOR_BINS


""" visible variables """

if OCCLUSION == 0:
    NUM_VISIBLE_CONF = 1  # S
    num_visible_state = NUM_VISIBLE_CONF * \
        NUM_DEPTH_LEVEL  # M = R * S 从 0 开始 m = r * S + s
    visibility_conf_mat = np.array([[1, 1]])
elif OCCLUSION == 1:
    NUM_VISIBLE_CONF = 2  # S
    num_visible_state = NUM_VISIBLE_CONF * \
        NUM_DEPTH_LEVEL  # M = R * S 从 0 开始 m = r * S + s
    visibility_conf_mat = np.array([[1, 1], [1, 0]])
else:
    print(f"Wrong occlusion: {OCCLUSION}")

## depth value for each level
depth_vec = [i for i in range(NUM_DEPTH_LEVEL)]


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

###############################


### b_mat, visible_state, disparity_image, visible_image

b_mat = np.zeros([HEIGHT * WIDTH, num_visible_state])
visible_state = np.zeros([HEIGHT, WIDTH])
disparity_image = np.zeros([HEIGHT, WIDTH])
# disparity_image = true_disparity_image.copy()
visible_image = np.zeros([HEIGHT, WIDTH])
