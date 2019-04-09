import tensorflow as tf
from tensorflow_time_two.python.ops import time_two_ops

x = tf.constant([1,2,3,4,5,6,7,8,9])
y = time_two_ops.time_two(x)

with tf.Session() as sess:
	print(sess.run(y))
