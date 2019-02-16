# vgg

- 不通过 stride = 2 的方式来减小 feature map size. On the contrary, it directly uses max-pooling to reduce the size of feature map.
- vgg 在 pool 的时候是如何增加 channel 数量的?
  - 不是在 pool 的时候增大 channel 数量的
  - 是在 pool 之后的卷基层直接卷回来的