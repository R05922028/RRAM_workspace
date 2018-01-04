from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from numpy import random

def error(x):
   return np.cos(x)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, shape=[None, 784])/ 255.
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ==Convolution layer== #
W_conv1	= tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1)) # 5x5, input_size=1, output_size=32
b_conv1 = tf.Variable(tf.zeros([32]))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME')
h_conv1 = tf.py_func(error, [h_conv1], tf.float32) # calculate error
h_conv1 = h_conv1 + b_conv1
h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32

W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b_conv2 = tf.Variable(tf.zeros(([64])))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME')
h_conv2 = tf.py_func(error, [h_conv2], tf.float32) # calculate error
h_conv2 = h_conv2 + b_conv2
h_conv2 = tf.nn.relu(h_conv2) # 14x14x64
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 7x7x64
# ==Fully connected layer== #
W_fcon1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fcon1 = tf.Variable(tf.zeros([1024]))
flatten = tf.reshape(h_pool2, [-1,7*7*64])
h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
h_fcon1 = tf.nn.relu(h_fcon1)
h_drop1 = tf.nn.dropout(h_fcon1, 0.5)

W_fcon2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fcon2 = tf.Variable(tf.zeros([10]))
h_fcon2 = tf.matmul(h_drop1, W_fcon2) + b_fcon2

prediction = h_fcon2

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, "./model/cnn_model.ckpt")

# Testing
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("accuracy: ",accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
