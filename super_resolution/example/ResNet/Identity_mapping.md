# Idendity mapping 阅读感想

$$
y_l = h(x_l) + F(x_l, W_l) \\
x_{l+1} = f(y_l)
$$

在 ResNet 提出的论文中，F 代表几个卷积层，h 是一个 identity mapping, f 是 ReLU function. 论文中提到一个 `information path` 的概念，在原始论文中提出的 `residual unit`，h 是一个 identity mapping, 但是 f 是一个 ReLU，所以还是无法达到 _clean information path_ 的地步，出于这个想法与目的，作者提出了 `pre-activation`，在论文的 4.1 节中我们也可以看到，`pre-activation` 与 `post-activation` 的内在关联性。