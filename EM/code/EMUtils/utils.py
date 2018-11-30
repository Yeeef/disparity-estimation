from EMUtils.config import *
import math
import numpy as np


def init_param_with_truth():
    for i in range(HEIGHT * WIDTH):
        row, col = map_1D_to_2D(i)
        disparity = disparity_image[row, col]
        if disparity == 0:
            disparity = np.random.randint(1, 20)
        r = depth_vec.index(disparity)
        s = 0
        m = map_r_s_to_m(r, s)
        b_mat[i, m] = 1
        # for s in range(NUM_VISIBLE_CONF):
        #     m = map_r_s_to_m(r, s)
        #     is_visible = visibility_conf_mat[s][1]
        #     kth_row, kth_col = row, col - disparity

        #     if is_visible:
        #         if is_out_image(kth_row, kth_col):
        #             color_prob = 0
        #         else:
        #             color_prob = norm_pdf(
        #                 I[1][kth_row, kth_col], ideal_image[row, col], covariance[0])
        #     else:
        #         if is_out_image(kth_row, kth_col):
        #             color_prob = 1
        #         else:
        #             color_prob = hist_prob(1, I[1][kth_row, kth_col])

        #     b_mat[i, m] = color_prob
        b_mat[i] = b_mat[i] / sum(b_mat[i])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = map_2D_to_1D(i, j)
            max_index = np.argmax(b_mat[index])
            visible_state[i, j] = max_index
            r, s = map_m_to_r_s(max_index)
            is_visible = visibility_conf_mat[s][1]
            visible_image[i, j] = is_visible


def init_param():
    for i in range(HEIGHT * WIDTH):
        for m in range(num_visible_state):
            row, col = map_1D_to_2D(i)
            r, s = map_m_to_r_s(m)
            disparity = depth_vec[r]
            ## 在第一幅图中能否看到
            is_visible = visibility_conf_mat[s][1]

            ## 在第一幅图中，根据 disparity 算出的坐标
            kth_row, kth_col = row, col - disparity

            if is_visible:
                if is_out_image(kth_row, kth_col):
                    color_prob = 0.0001
                else:
                    color_prob = norm_pdf(
                        I[1][kth_row, kth_col], ideal_image[row, col], covariance[0])
            else:
                # color_prob = hist_prob(1, I[0][row, col])
                if is_out_image(kth_row, kth_col):
                    color_prob = hist_prob(0, I[0][row, col])
                else:
                    color_prob = hist_prob(1, I[1][kth_row, kth_col])

            b_mat[i, m] = color_prob
        ## 归一化
        if sum(b_mat[i]) > 0:
            b_mat[i] = b_mat[i] / sum(b_mat[i])
        else:
            b_mat[i] = [1 / num_visible_state for _ in range(num_visible_state)]

    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = map_2D_to_1D(i, j)
            max_index = np.argmax(b_mat[index])
            visible_state[i, j] = max_index
            r, s = map_m_to_r_s(max_index)
            disparity = depth_vec[r]
            is_visible = visibility_conf_mat[s][1]
            disparity_image[i, j] = disparity
            visible_image[i, j] = is_visible

    # disparity_image = true_disparity_image.copy()


def map_ideal_to_kth_with_disparity(row, col, k, disparity):
    if k == 0:
        return row, col
    else:
        return row, col - disparity
# def map_ideal_to_kth(i, j, k, disparity_image):
#     if k == 0:
#         return int(i), int(j)
#     else:
#         disp = disparity_image[i, j]
#         return int(i), int(j - disp)

## @param i: row(0 ~ HEIGHT - 1)
## @param j: col(0 ~ WIDTH - 1)
## @return: 1d index


def map_2D_to_1D(i, j, n_rows=HEIGHT, n_cols=WIDTH):
    if j >= n_cols:
        print(f"[map_2D_to_1D]: width {j} index out of range")
        return None
    elif j < 0:
        print(f"[map_2D_to_1D]: width {j} index out of range")
    if i >= n_rows:
        print(f"[map_2D_to_1D]: height {i} index out of range")
        return None
    elif i < 0:
        print(f"[map_2D_to_1D]: height {i} index out of range")
    return int(i * n_cols + j)

## @param index: 1d index
## @return i: row(0 ~ HEIGHT - 1)
## @return j: col(0 ~ WIDTH - 1)


def map_1D_to_2D(index, n_rows=HEIGHT, n_cols=WIDTH):
    i = index // n_cols
    j = index - i * n_cols
    if i >= n_rows:
        print(f"[map_1D_to_2D] height {i} index out of range")
        return None
    if j >= n_cols:
        print(f"[map_1D_to_2D] width {j} index out of range")
        return None
    return (int(i), int(j))


def map_m_to_r_s(m, R=NUM_DEPTH_LEVEL, S=NUM_VISIBLE_CONF):
    return map_1D_to_2D(m, R, S)


def map_r_s_to_m(r, s, R=NUM_DEPTH_LEVEL, S=NUM_VISIBLE_CONF):
    return map_2D_to_1D(r, s, R, S)


def is_out_image(i, j, height=HEIGHT, width=WIDTH):
    res = False
    if i < 0 or i >= height or j < 0 or j >= width:
        res = True

    return res


def norm_pdf(x, mu, sigma):
    return math.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (math.sqrt(2 * math.pi) * sigma)


def hist_prob(k, color, num_color_in_bin=num_color_in_bin):
    bin_index = int(color // num_color_in_bin)
    return hist_mat[k][bin_index]


def neighbor(index, height=HEIGHT, width=WIDTH):
    i, j = map_1D_to_2D(index)
    left = i * width + j - 1
    right = i * width + j + 1
    upper = (i - 1) * width + j
    lower = (i + 1) * width + j

    if i == 0:
        if j == 0:
            return (lower, right)
        elif j == width - 1:
            return (lower, left)
        else:
            return (lower, left, right)

    if i == height - 1:
        if j == 0:
            return (upper, right)
        elif j == width - 1:
            return (upper, left)
        else:
            return(upper, left, right)

    if j == 0:
        return (upper, lower, right)
    if j == width - 1:
        return (upper, lower, left)

    return (left, right, upper, lower)

def nearest_int(num):
    floor = int(num)
    ceil = floor + 1
    if num - floor < 0.5:
        return floor
    else:
        return ceil
