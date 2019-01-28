
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
 
device_name="/cpu:0"
#device_name="/gpu:0"
 
shape=(int(10000),int(10000))
 
with tf.device(device_name):
#形状为shap,元素服从minval和maxval之间的均匀分布
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)
 
    startTime = datetime.now()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)
 
print("\n" * 2)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)
