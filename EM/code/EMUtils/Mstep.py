from EMUtils.config import *
from . import utils
import math


## 已知 visible variable
def update_one_pixel(index):
    r, s = utils.map_m_to_r_s(visible_state[utils.map_1D_to_2D(index)])
    deno = 0
    numer = 0
    for k in range(NUM_INPUT):
        row, col = utils.map_1D_to_2D(index)
        kth_row, kth_col = utils.map_ideal_to_kth(row, col, k, disparity_image)
        numer += I[k][kth_row, kth_col] * visibility_conf_mat[s][k]
        deno += visibility_conf_mat[s][k]

    ideal_image[row, col] = numer / deno


def update_y_ideal():
    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = utils.map_2D_to_1D(i, j)
            update_one_pixel(index)


def update_cov():
    deno = 0
    numer = 0
    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = utils.map_2D_to_1D(i, j)
            for k in range(NUM_INPUT):
                r, s = utils.map_m_to_r_s(visible_state[i, j])
                kth_row, kth_col = utils.map_ideal_to_kth(i, j, k, disparity_image)
                tmp = I[k][kth_row, kth_col] - ideal_image[i, j]
                numer += visibility_conf_mat[s][k] * (tmp ** 2)
                deno += visibility_conf_mat[s][k]
    # if numer / deno > 10:
    #     covariance[0] = 10
    # else:
    #     covariance[0] = numer / deno
    covariance[0] = numer / deno
    print(covariance[0])


def is_in_color_bin(color, b, num_color_in_bin=num_color_in_bin):
    bin_index = color // num_color_in_bin
    if int(bin_index) == b:
        return True
    else:
        return False


def update_h_of_one_image(k):
    ## b: 0 ~ 31
    for b in range(NUM_COLOR_BINS):
        tmp = 0
        for i in range(HEIGHT):
            for j in range(WIDTH):
                r, s = utils.map_m_to_r_s(visible_state[i, j])
                kth_row, kth_col = utils.map_ideal_to_kth(i, j, k, disparity_image)
                if is_in_color_bin(I[k][kth_row, kth_col], b) and visibility_conf_mat[s][k] == 0:
                    tmp += 1
        hist_mat[k, b] = tmp

    # 归一化
    if sum(hist_mat[k]) == 0:
        hist_mat[k] = np.zeros([1, NUM_COLOR_BINS]) / NUM_COLOR_BINS
    else:
        hist_mat[k] = hist_mat[k] / (sum(hist_mat[k]) * 10)  # 标准化


def update_hist():
    for k in range(NUM_INPUT):
        update_h_of_one_image(k)


def M_step():
    update_y_ideal()
    update_cov()
    update_hist()
    

