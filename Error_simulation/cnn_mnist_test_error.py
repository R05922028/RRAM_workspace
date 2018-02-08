from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from random import *
from math import *

def isround(p):
    if uniform(0,1) >= p:
        return 0
    else:
        return 1

def stochasticRounding(x, FL):
    power = 1 << FL
    tmp = floor(x*power)
    floor_data = tmp / power
    prob = (x - floor_data) * power
    return isround(prob) * (1/power) + floor_data

def convert(x, IL, FL):
    maximum = (1 << (IL-1)) - 1/float(1<<FL)
    minimum = -1 * (1<<(IL-1))
    if x >= maximum:
        return maximum
    elif x <= minimum:
        return minimum
    else:
        return stochasticRounding(x, FL)

def convert2TwosComplement(x, FL, WL):
    power = 1 << FL
    x = int(x * power)
    binstr = np.binary_repr(x, width=WL)
    arr = list(map(int, list(binstr)))
    return arr

def roundAndConvert(x, IL, FL, WL):
    x_round = convert(x, IL, FL)
    arr = convert2TwosComplement(x_round, FL, WL)
    return arr

def f(x, IL, FL, WL):
    xshape = x.shape 
    x = [roundAndConvert(i, IL, FL, WL) for i in np.nditer(x)]
    x = np.array(x).T 
    return np.float32(x).reshape(((-1,) + xshape))

IL = 3
FL = 3
WL = IL + FL

'''
W = tf.Variable(np.arange(12).reshape((2,2,1,3)))
W = tf.py_func(f, [W, IL, FL, WL], tf.int8)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print(sess.run(W))
print('--------------')
print(sess.run(W[0]))
print(sess.run(W[1]))
print(sess.run(W[2]))
print(sess.run(W[3]))

'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ==Convolution layer== #
with tf.name_scope('Conv1'):
    with tf.name_scope('Weights'):
        W_conv1	= tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1)) # 5x5, input_size=1, output_size=32
        W_conv1 = tf.py_func(f, [W_conv1, IL, FL, WL], tf.float32) 
    with tf.name_scope('Biases'):
        b_conv1 = tf.Variable(tf.zeros([32]))
    with tf.name_scope('Convolution'):
        #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        h_conv1_arr = []
        for i in range(WL):
            h_conv1_arr.append(tf.nn.conv2d(x_image, W_conv1[i], strides=[1,1,1,1], padding='SAME'))
        '''
        h_conv1_0 = tf.nn.conv2d(x_image, W_conv1[0], strides=[1,1,1,1], padding='SAME')
        h_conv1_1 = tf.nn.conv2d(x_image, W_conv1[1], strides=[1,1,1,1], padding='SAME')
        h_conv1_2 = tf.nn.conv2d(x_image, W_conv1[2], strides=[1,1,1,1], padding='SAME')
        h_conv1_3 = tf.nn.conv2d(x_image, W_conv1[3], strides=[1,1,1,1], padding='SAME')
        '''
        sig = IL - 1
        h_conv1 = -h_conv1_arr[0]*(2**sig)
        
        
        for i in range(1,WL):
            sig -= 1
            h_conv1 += h_conv1_arr[i] * (2**sig)
        
    with tf.name_scope('Relu'):
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
with tf.name_scope('Maxpooling'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32

with tf.name_scope('Conv2'):
    with tf.name_scope('Weights'):
        W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
    with tf.name_scope('Biases'):
        b_conv2 = tf.Variable(tf.zeros(([64])))
    with tf.name_scope('Convolution'):
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    with tf.name_scope('Relu'): 
        h_conv2 = tf.nn.relu(h_conv2) # 14x14x64
with tf.name_scope('Maxpooling'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 7x7x64
# ==Fully connected layer== #
with tf.name_scope('Dense1'):
    with tf.name_scope('Weights'):
        W_fcon1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
    with tf.name_scope('Biases'):
        b_fcon1 = tf.Variable(tf.zeros([1024]))
    with tf.name_scope('Flatten'):
        flatten = tf.reshape(h_pool2, [-1,7*7*64])
    with tf.name_scope('Formula'):
        h_fcon1 = tf.matmul(flatten, W_fcon1) + b_fcon1
    with tf.name_scope('Relu'):
        h_fcon1 = tf.nn.relu(h_fcon1)
    with tf.name_scope('Dropout'):
        h_drop1 = tf.nn.dropout(h_fcon1, 0.5)

W_fcon2 = tf.Variable(tf.zeros([1024,10]), name='w')
b_fcon2 = tf.Variable(tf.zeros([10]), name='b')
with tf.name_scope('Dense2'):
    with tf.name_scope('Formula'):
        h_fcon2 = tf.matmul(h_drop1, W_fcon2) + b_fcon2
        prediction = h_fcon2

prediction = tf.identity(prediction, name='prediction')


# initialize Graph
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, "./model/cnn_model.ckpt")

# Testing
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)

print("accuracy: ",accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

