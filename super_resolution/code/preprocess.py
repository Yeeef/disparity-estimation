import numpy as np
import h5py
from PIL import Image
import os
from matplotlib import pyplot as plt
import pickle


DATA_PREFIX = '/Users/yee/Downloads/NYUv2pics'
OUT_PREFIX = '/Users/yee/Desktop/NYUv2'

for i in range(1449):
    depth = np.array(
        Image.open(
            os.path.join(DATA_PREFIX, 'nyu_depths', f'{i}.png')
        )
    )
    img = np.array(
        Image.open(
            os.path.join(DATA_PREFIX, 'nyu_images', f'{i}.jpg')
        )
    )
    # channels_first
    img = np.transpose(img, [2, 0, 1])
    depth = np.expand_dims(depth, 0)
    with open(os.path.join(OUT_PREFIX, f'{i}.pickle'), 'wb') as f:
        pickle.dump(np.concatenate([img, depth]), f)

# depth = np.array(Image.open(
#     os.path.join(DATA_PREFIX, 'nyu_depths', '0.png')
# ))

# img = np.array(Image.open(
#     os.path.join(DATA_PREFIX, 'nyu_images', '0.jpg')
# ))
# img = np.transpose(img, [2, 0, 1])
# depth = np.expand_dims(depth, 0)
# _, height, width = depth.shape
# print(np.concatenate([img, depth]).shape)

# drgb = np.concatenate([img, depth])

# # with open('test.pickle', 'wb') as f:
# #     pickle.dump(drgb, f)

# with open('test.pickle', 'rb') as f:
#     drgb = pickle.load(f)

# # Image.fromarray(np.transpose(drgb, [1, 2, 0])).save('test.png')
# plt.figure()
# plt.imshow(np.transpose(drgb[:3], [1, 2, 0]))
# plt.figure()
# plt.imshow(np.reshape(drgb[3], [height, width]))
# plt.show()




