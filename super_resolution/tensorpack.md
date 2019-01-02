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

## trainer

## TowerTrainer

- what is Tower Function
  - a tower function is a callable that takes input tensors and adds **one replicate** of the model to the graph.

- MultiGPU Trainers
  - n each iteration, instead of taking one tensor for all GPUs and split, all GPUs take tensors from the `InputSource`. So the total batch size across all GPUs would become `(batch size of InputSource) * #GPU`.
  - The tower function (your model code) will get called multipile times on each GPU. You must follow the abovementieond rules of tower function.

## Training Interface

## Callbacks

Callback is an interface to do __everything__ else besides the training iterations.

## Inference

Tensorpack is a training interface – __it doesn’t care what happened after training.__ You already have everything you need for inference or model diagnosis after training:

Therefore, you can build the graph for inference, load the checkpoint, and apply any processing or deployment TensorFlow supports. These are unrelated to tensorpack, and you’ll need to read TF docs and **do it on your own.**

### Step 1: build the model

### Step 2: load the checkpoint

## 还需理解的地方

- tower
- rms
- build_graph rules
  - 在 build_graph 中可以调用 `tensorpack` 中 `summary` 方法
- op


