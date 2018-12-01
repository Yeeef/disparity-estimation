from EMUtils.config import *
from . import utils
import math
from functools import reduce




## depth level 的距离
def cal_D1(r, p, sigma_d=sigma_d):
    D1 = abs(r - p) / (NUM_DEPTH_LEVEL * sigma_d)
    return D1

## visibility level 的距离
## number of dissimilar entries of v_i^s and v_j^q
## s, q configuration 的index，可以对应出两个向量



def cal_D2_fast(s, q, sigma_v=sigma_v):
    def fun(pair):
        if pair[0] == pair[1]:
            return 0
        else:
            return 1
    s_conf = visibility_conf_mat[s]
    q_conf = visibility_conf_mat[q]
    return (reduce(lambda x, y: x + y, map(fun, zip(s_conf, q_conf)))) / sigma_v
    



def psi_mn(m, n, C=C):
    r, s = utils.map_m_to_r_s(m)
    p, q = utils.map_m_to_r_s(n)
    return (math.exp(-cal_D1(r, p, sigma_d) - cal_D2_fast(s, q, sigma_v)) + C)

def update_bim_iterunit(i, m, k):
    r, s = utils.map_m_to_r_s(m)
    ## 获得 visibility 配置
    conf_s = visibility_conf_mat[s]
    is_visible = conf_s[k]
    disparity = depth_vec[r]
    row, col = utils.map_1D_to_2D(i)

    kth_row, kth_col = utils.map_ideal_to_kth_with_disparity(row, col, k, disparity)
    # if utils.is_out_image(kth_row, kth_col):
    #     if k == 1 and is_visible:
    #         return 0.0001
    #     else:
    #         return 1
    if is_visible:
        ## 正态分布
        if utils.is_out_image(kth_row, kth_col):
            return 0.0001

        color_prob = utils.norm_pdf(I[k][kth_row, kth_col],
                              ideal_image[row, col], covariance[0])
    else:
        ## 假定在第一幅图中一定不出现看不见的点
        ## 从 hist 求概率
        # color_prob = utils.hist_prob(k, I[0][row, col])
        if utils.is_out_image(kth_row, kth_col):
            color_prob = utils.hist_prob(0, I[0][row, col])
        else:
            color_prob = utils.hist_prob(1, I[1][kth_row, kth_col])
        # if utils.is_out_image(kth_row, kth_col):
        #     return 0.0001
        # color_prob = utils.hist_prob(k, I[k][kth_row, kth_col])

    return color_prob


def update_bim_fast(i, m):
    res = 1
    for k in range(NUM_INPUT):
        res = res * update_bim_iterunit(i, m, k)

    ## 第二部分的概率
    log_psi = [math.log(psi_mn(m, n, C)) for n in range(num_visible_state)]
    tmp_mat = np.vstack([b_mat[j] for j in utils.neighbor(i)])
    summation = np.sum(np.dot(tmp_mat, log_psi))

    
    b_mat[i, m] = res * math.exp(summation)

def update_bi_fast(i):
    [update_bim_fast(i, m) for m in range(num_visible_state)]

    if sum(b_mat[i]) > 0:
            b_mat[i] = b_mat[i] / sum(b_mat[i])
    else:
       b_mat[i] = [1 / num_visible_state for _ in range(num_visible_state)]

    row, col = utils.map_1D_to_2D(i)
    if col == WIDTH - 1:
        print("\r" + str(row), end="")
    

def update_b_fast():
    cnt = 0
    [update_bi_fast(i) for i in range(HEIGHT * WIDTH)]
 


def update_visible_argmax_i_fast_fun(i, j, max_index):
    visible_state[i, j] = max_index
    r, s = utils.map_m_to_r_s(max_index)
    disparity = depth_vec[r]
    is_visible = visibility_conf_mat[s][1]            
    disparity_image[i, j] = disparity
    visible_image[i, j] = is_visible

def update_visible_argmax_i_fast(i, b_mat=b_mat):
    [update_visible_argmax_i_fast_fun(i, j, np.argmax(b_mat[utils.map_2D_to_1D(i, j)])) for j in range(WIDTH)]


def update_visible_argmax_fast(b_mat=b_mat):
    [update_visible_argmax_i_fast(i, b_mat) for i in range(HEIGHT)]
    

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
            
            
            
            nearest_disparity = utils.nearest_int(disparity)
            nearest_visibility = utils.nearest_int(visibility)
            visible_image[i, j] = nearest_visibility
            disparity_image[i, j] = nearest_disparity
            new_r = depth_vec.index(nearest_disparity)
            if nearest_visibility == 1:
                new_s = 0
            else:
                new_s = 1
            
            visible_state[i, j] = utils.map_r_s_to_m(new_r, new_s)

def E_step_fast():
    update_b_fast()
    # update_visible_expectation()
    update_visible_argmax_fast()


def free_energy():
    sum1 = 0
    for i in range(HEIGHT):
        for j in range(WIDTH):
            for k in range(NUM_INPUT):
                for m in range(num_visible_state):
                    r, s = utils.map_m_to_r_s(visible_state[i, j])
                    disparity = depth_vec[r]
                    kth_row, kth_col = utils.map_ideal_to_kth_with_disparity(i, j, k, disparity)
                    
                    conf_s = visibility_conf_mat[s]
                    is_visible = conf_s[k]
                    if is_visible:
                        prob = utils.norm_pdf(I[k][kth_row, kth_col],
                                        ideal_image[i, j], covariance[0])
                    else:
                        prob = utils.hist_prob(k, I[k][kth_row, kth_col])

                    sum1 += b_mat[utils.map_2D_to_1D(i, j), m] * math.log(prob)
    print(f"sum1: {sum1}")
    
    sum3 = 0
    for row in range(HEIGHT):
        for col in range(WIDTH):
            index = utils.map_2D_to_1D(row, col)
            for m in range(num_visible_state):
                sum3 += b_mat[index, m] * math.log(b_mat[index, m])
    print(f"sum3: {sum3}")

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
    
    return -sum1 - sum2 + sum3
