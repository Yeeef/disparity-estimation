import numpy as np
from numpy import linalg

# input color images
# I_j[X] specify the color of the projection pixel in image j
I = []

# computed depth image
# D_j[X] specify the computed depth of the projection pixel in image j
D = []

# actual depth
d = []

# computed color images, it is kinda ideally the actual color of the 3d
# point corresponding to the image pixel
# C[X] specify the actual color of the 3D point
C = []

# the list of the projection matrices
# shape = 3 * 4
P = []

# list of visible dict
# visible dict, whether point X is visible in the image i
V = []



## functions

## Calculate the 3d coordinate from the pixel x in image
## and its depth d, with the rotation R, translation T
## and intrinsic matrix K
def cal_3dpoint_from_(x, d, K, R, T):
    return (d * linalg.inv(K @ R) @ x - np.transpose(R) @ T)




