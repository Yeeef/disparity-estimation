# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('lena.png')
plt.figure('lena')
plt.imshow(img)
plt.show()
print img


