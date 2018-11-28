import utils
import E_utils
import math


def update_bim_iterunit(i, m, k, r, s, visibility_conf_mat):
    ## 获得 visibility 配置
    conf_s = visibility_conf_mat[s]
    is_visible = conf_s[k]
    row, col = utils.map_1D_to_2D(i)
    kth_row, kth_col = utils.map_ideal_to_kth(row, col)
    if utils.is_out_image(kth_row, kth_col):
        return 1
#     print(kth_row, kth_col)
#     print(k)
#     print(row, col)
#     print(I[k][kth_row, kth_col])
#     print(ideal_image[row, col])
    if is_visible:
        ## 正态分布
        color_prob = utils.norm_pdf(I[k][kth_row, kth_col],
                              ideal_image[row, col], covariance)
    else:
        ## 从 hist 求概率
        color_prob = E_utils.hist_prob(k, I[k][kth_row, kth_col])

    ## 第二部分的概率
    summation = 0
    for j in E_utils.neighbor(i):
        for n in range(num_visible_state):
            summation += b_mat_last[j, n] * math.log(psi_mn(m, n, C))

    return color_prob * math.exp(summation)
