import math
import utils

## depth level 的距离
HEIGHT = 375
WIDTH = 450
sigma_d = 300
sigma_v = 30
NUM_DEPTH_LEVEL = 10
C = 10 ** (-4)
NUM_COLOR_BINS = 32
num_color_in_bin = 256 / NUM_COLOR_BINS

def cal_D1(r, p, num_depth_level=NUM_DEPTH_LEVEL, sigma_d=sigma_d):
    D1 = abs(r - p) / (num_depth_level * sigma_d)
    return D1

## visibility level 的距离
## number of dissimilar entries of v_i^s and v_j^q
## s, q configuration 的index，可以对应出两个向量


def cal_D2(s, q, visibility_conf_mat, sigma_v=sigma_v):
    s_conf = visibility_conf_mat[s]
    q_conf = visibility_conf_mat[q]
    dissimilar_count = 0
    for i in range(len(s_conf)):
        if(s_conf[i] != q_conf[i]):
            dissimilar_count += 1
    return dissimilar_count / sigma_v

## constant C


def psi_mn(m, n, C=C, sigma_d=sigma_d, sigma_v=sigma_v):
    r, s = utils.map_m_to_r_s(m)
    p, q = utils.map_m_to_r_s(n)
    return (math.exp(-cal_D1(r, p, sigma_d) - cal_D2(s, q, sigma_v)) + C)



def neighbor(index, height=HEIGHT, width=WIDTH):
    i, j = utils.map_1D_to_2D(index)
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


def norm_pdf(x, mu, sigma):
    return math.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (math.sqrt(2 * math.pi) * sigma)

def hist_prob(k, color, hist_mat, num_color_in_bin=num_color_in_bin):
    bin_index = int(color // num_color_in_bin)
    return hist_mat[k][bin_index]
