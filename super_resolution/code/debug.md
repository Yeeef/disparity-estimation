# debug record

## reminder

- `Prefetch` 暂时注释
- `QueueInput` 必须配以 ds 特殊的 `__iter__`, 简单的 `yield` 不能使用这个接口
- `Augmentor` 在使用中不能出现 tf ops, 这是 _the graph is finalized_
- `del` 做什么的？
- tensorpack 在定义 inputs 的过程中，默认是一个 list: 如 `[img, label]`, 看它的源码 `mnist.py`, `cifar.py` 也是一样的原理，所以我必须在 `data.py` 上做一些改动，在 `__iter__` 中 ` yield [self.data[k]]` 即可，搞了我一晚上的问题，fuck