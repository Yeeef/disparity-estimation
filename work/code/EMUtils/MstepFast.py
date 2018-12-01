
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
        disparity = depth_vec[r]
        kth_row, kth_col = utils.map_ideal_to_kth_with_disparity(row, col, k, disparity)
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
    tmp_sum = 0
    s_0 = 0
    s_1 = 0
    for i in range(HEIGHT):
        for j in range(WIDTH):
            for k in range(NUM_INPUT):
                r, s = utils.map_m_to_r_s(visible_state[i, j])
                disparity = depth_vec[r]
                kth_row, kth_col = utils.map_ideal_to_kth_with_disparity(i, j, k, disparity)
                tmp = I[k][kth_row, kth_col] - ideal_image[i, j]
                numer += visibility_conf_mat[s][k] * (tmp ** 2)
                deno += visibility_conf_mat[s][k]
    covariance[0] = numer / deno
    print(f"covariance: {covariance[0]}")


def is_in_color_bin(color, b, num_color_in_bin=num_color_in_bin):
    bin_index = color // num_color_in_bin
    if int(bin_index) == b:
        return True
    else:
        return False

def update_hist_fast():
    for i in range(HEIGHT):
        for j in range(WIDTH):
            r, s = utils.map_m_to_r_s(visible_state[i, j])
            disparity = depth_vec[r]
            kth_row, kth_col = utils.map_ideal_to_kth_with_disparity(i, j, 1, disparity)
            if visibility_conf_mat[s][1] == 0:
                if not utils.is_out_image(kth_row, kth_col):
                    bin_index = I[1][kth_row, kth_col] // num_color_in_bin
                    hist_mat[1, int(bin_index)] += 1
                else:
                    bin_index = I[0][i, j] // num_color_in_bin
                    hist_mat[0, int(bin_index)] += 1
    for k in range(NUM_INPUT):
        if sum(hist_mat[k]) == 0:
            hist_mat[k] = np.zeros([1, NUM_COLOR_BINS]) / NUM_COLOR_BINS
        else:
            hist_mat[k] = hist_mat[k] / (sum(hist_mat[k]))  # 标准化


def M_step_fast():
    update_y_ideal()
    update_cov()
    update_hist_fast()
    

