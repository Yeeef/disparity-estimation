from EMUtils.config import *
from . import utils
import math


# class EStep(object):
#     def __init__(self):


## depth level 的距离
def cal_D1(r, p, sigma_d=sigma_d):
    D1 = abs(r - p) / (NUM_DEPTH_LEVEL * sigma_d)
    return D1

## visibility level 的距离
## number of dissimilar entries of v_i^s and v_j^q
## s, q configuration 的index，可以对应出两个向量


def cal_D2(s, q, sigma_v=sigma_v):
    s_conf = visibility_conf_mat[s]
    q_conf = visibility_conf_mat[q]
    dissimilar_count = 0
    for i in range(len(s_conf)):
        if(s_conf[i] != q_conf[i]):
            dissimilar_count += 1
    return dissimilar_count / sigma_v



def psi_mn(m, n, C=C):
    r, s = utils.map_m_to_r_s(m)
    p, q = utils.map_m_to_r_s(n)
    return (math.exp(-cal_D1(r, p, sigma_d) - cal_D2(s, q, sigma_v)) + C)


def update_bim_iterunit(i, m, k):
    r, s = utils.map_m_to_r_s(m)
    ## 获得 visibility 配置
    conf_s = visibility_conf_mat[s]
    is_visible = conf_s[k]
    row, col = utils.map_1D_to_2D(i)
    kth_row, kth_col = utils.map_ideal_to_kth(row, col, k, disparity_image)
    if utils.is_out_image(kth_row, kth_col):
        if k == 1 and is_visible:
            return 0
        else:
            return 1
    if is_visible:
        ## 正态分布

        color_prob = utils.norm_pdf(I[k][kth_row, kth_col],
                              ideal_image[row, col], covariance[0])
    else:
        ## 从 hist 求概率
        color_prob = utils.hist_prob(k, I[k][kth_row, kth_col])

    return color_prob


def update_bim(i, m):
    res = 1
    for k in range(NUM_INPUT):
        res = res * update_bim_iterunit(i, m, k)
    # print(i, m, res)

    ## 第二部分的概率
    summation = 0
    for j in utils.neighbor(i):
        for n in range(num_visible_state):
            summation += b_mat[j, n] * math.log(psi_mn(m, n, C))
    b_mat[i, m] = res * math.exp(summation)


def update_b():
    cnt = 0
    for i in range(HEIGHT * WIDTH):
        cnt += 1
        row, col = utils.map_1D_to_2D(i)

        if cnt == WIDTH:
            cnt = 0
            print("\r" + str(row), end="")
            ## break

        r, s = utils.map_m_to_r_s(visible_state[row, col])
        for m in range(num_visible_state):

            #             print(f"[update_b]: i = {i}")
            update_bim(i, m)

    ## 归一化
    for index in range(HEIGHT * WIDTH):
        b_mat[index] = b_mat[index] / sum(b_mat[index])


def update_visible(b=b_mat):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = utils.map_2D_to_1D(i, j)
            max_index = np.argmax(b[index])
            visible_state[i, j] = max_index
            r, s = utils.map_m_to_r_s(max_index)
            disparity = depth_vec[r]
            is_visible = visibility_conf_mat[s][1]
            disparity_image[i, j] = disparity
            visible_image[i, j] = is_visible

def update_visible_expectation(b=b_mat):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            index = utils.map_2D_to_1D(i, j)
            disparity = 0
            visibility = 0
            for m in range(num_visible_state):
                r, s = utils.map_m_to_r_s(m)
                disparity += b[index, m] * depth_vec[r]
                visibility += b[index, m] * visibility_conf_mat[s][1]
            
            disparity_image[i, j] = disparity
            
            nearest_disparity = utils.nearest_int(disparity)
            nearest_visibility = utils.nearest_int(visibility)
            visible_image[i, j] = nearest_visibility
            new_r = depth_vec.index(nearest_disparity)
            if nearest_visibility == 1:
                new_s = 0
            else:
                new_s = 1
            
            visible_state[i, j] = utils.map_r_s_to_m(new_r, new_s)

def E_step():
    update_b()
    update_visible_expectation()


def free_energy():
    sum1 = 0
    for i in range(HEIGHT):
        for j in range(WIDTH):
            for k in range(NUM_INPUT):
                for m in range(num_visible_state):
                    kth_row, kth_col = utils.map_ideal_to_kth(i, j, k, disparity_image)
                    r, s = utils.map_m_to_r_s(visible_state[i, j])
                    conf_s = visibility_conf_mat[s]
                    is_visible = conf_s[k]
                    if is_visible:
                        prob = utils.norm_pdf(I[k][kth_row, kth_col],
                                        ideal_image[i, j], covariance[0])
                    else:
                        prob = utils.hist_prob(k, I[k][kth_row, kth_col])

                    sum1 += b_mat[utils.map_2D_to_1D(i, j), m] * prob
    print(f"sum1: {sum1}")
    sum2 = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            index = utils.map_2D_to_1D(row, col)
            for j in utils.neighbor(index):
                for m in range(num_visible_state):
                    for n in range(num_visible_state):
                        sum2 += b_mat[index, m] * b_mat[j, n] * \
                            math.log(psi_mn(m, n, C))
    print(f"sum2: {sum2}")
    sum3 = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            index = utils.map_2D_to_1D(row, col)
            for m in range(num_visible_state):
                sum3 += b_mat[index, m] * math.log(b_mat[index, m])
    print(f"sum3: {sum3}")
    return -sum1 - sum2 + sum3
