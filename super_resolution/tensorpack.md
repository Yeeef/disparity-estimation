# TensorPack learning

> Tensorpack common API learning with MNIST toy example.

## DataFlow

DataFlow is a library to build Python iterators for efficient data loading.
DataFlow is **independent of TensorFlow**.
`InputSource` interface links DataFlow to the graph for training.

- `__iter__()` generator method
- `__len__()` method

## Input Pipeline

> How to read data efficiently to work with TensorFlow

- Prepare Data in Parallel

### InputSource

> An abstract interface used by tensorpack trainers, to describe where the inputs come from and how they enter the graph

`DataFlow + QueueInput` will be a good choice.

## Symbolic Layers

- `argscope` gives you a context with default arguments.
- `LinearWrap` is a syntax sugar to simplify building "linear structure"

### some issues

- Regularizations may be handled differently: in tensorpack, users need to 