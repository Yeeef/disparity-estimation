# thoughts about resnet

## original paper

In original paper, the downsampling is done by a strided conv layer with stride's size of 2. 1 * 1 conv is used to match the dimension between residual and identity mapping. We can also use `tf.pad` to simply pad zeros to match the dimension as proposed in `resnet cifar10` in tensorpack example.

## Identity Mapping in Deep Residual Networks

- `pre-activation`: In this paper, they proposed a new residual unit, where BN and ReLU are put before the weight,