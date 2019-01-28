import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.dataflow import dataset

IMAGE_SIZE = 28

class Model(ModelDesc):
    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                tf.placeholder(tf.int32, (None, ), 'label')]

    def build_graph(self, image, label):
        """
        This function should build the model which takes the input variables
        and return cost at the end
        """
        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        
        image = tf.expand_dims(image, 3)
        image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3

        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu, filters=32):
            logits = (LinearWrap(image)
                      .Conv2D('conv0')
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv1')
                      .Conv2D('conv2')
                      .MaxPooling('pool1', 2)
                      .Conv2D('conv3')
                      .FullyConnected('fc0', 512, activation=tf.nn.relu)
                      .Dropout('dropout', rate=0.5)
                      .FullyConnected('fc1', 10, activation=tf.identity)())
            
            # a vector of length B with loss of each sample
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
            # the average cross-entropy loss
            cost = tf.reduce_mean(cost, name='cross_entropy_loss')

            correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
            accuracy = tf.reduce_mean(correct, name='accuracy')
            
            # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
            # 1. written to tensorboard
            # 2. written to stat.json
            # 3. printed after each epoch

            train_error = tf.reduce_mean(1 - correct, name='train_error')
            summary.add_moving_summary(train_error, accuracy)

            # Use a regex to find parameters to apply weight decay.
            # Here we apply a weight decay on all W (weight matrix) of all fc layers
            # If you don't like regex, you can certainly define the cost in any other methods.
            # 利用 L2 对 fc 系数进行正则化，权重衰减
            # 1e-5 为 \lambda
            wd_cost = tf.multiply(1e-5,
                                  regularize_cost('fc.*/W', tf.nn.l2_loss),
                                  name='regularize_cost')
            # 这里 add_n 其实只是为了保险? 之前 reduce_mean 的时候应该已经将cost 变成一个 scalar 了
            total_cost = tf.add_n([wd_cost, cost], name='total_cost')
            summary.add_moving_summary(cost, wd_cost, total_cost)

            # monitor histogram of all weight (of conv and fc layers) in tensorboard
            summary.add_param_summary(('.*/W', ['histogram', 'rms']))
            # the function should return the total cost to be optimized
            return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
                learning_rate=1e-3,
                global_step=get_global_step_var(),
                decay_steps=468 * 10,
                decay_rate=0.3,
                staircase=True,
                name='learning_rate'
                )

        tf.summary.scalar('lr', lr)

        return tf.train.AdamOptimizer(lr)

def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    # 若 remainder 为 True, 剩余不足256大小的样本也会成为一个小 batch
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)
    # Behave like an identity mapping, but print shape and 
    # range of the first few datapoints.
    train = PrintData(train)

    return train, test

if __name__ == '__main__':
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data()
    steps_per_epoch = len(dataset_train)

    # get the config which contains everything necessary in a training
    # A collection of options to be used for single-cost trainers.
    # Note that you do not have to use TrainConfig. 
    # You can use the API of Trainer directly, 
    # to have more fine-grained control of the training.
    config = TrainConfig(
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        # Input by iterating over a DataFlow and feed datapoints.
        data=FeedInput(dataset_train),
        callbacks=[
            ModelSaver(), # save the model after every epoch
            InferenceRunner(  # run inference(for validation) after every epoch
                dataset_test, # the DataFlow instance used for validation
                # Statistics of some scalar tensor. 
                # The value will be averaged over all given datapoints.
                ScalarStats(  # produce `val_accuracy` and `val_cross_entropy_loss`
                    ['cross_entropy_loss', 'accuracy'], prefix='val'
                )

            ),
            MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=10,

    )
    launch_train_with_config(config, SimpleTrainer())


