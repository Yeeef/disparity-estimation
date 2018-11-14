import numpy as np

HEIGHT = 375
WIDTH = 450
NUM_VISIBLE_CONF = 2  # S
NUM_DEPTH_LEVEL = 10  # R depth level

def map_ideal_to_kth(i, j, disparity_image):
    disp = disparity_image[i, j]
    return int(i), int(j - disp)

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


