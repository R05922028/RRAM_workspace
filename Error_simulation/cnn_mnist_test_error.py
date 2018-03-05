from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
from random import *
from math import *
from tqdm import tqdm

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

def decomposition(x, IL, FL, WL):
    xshape = x.shape 
    x = [roundAndConvert(i, IL, FL, WL) for i in np.nditer(x)]
    x = np.array(x).T 
    return np.float32(x).reshape(((-1,) + xshape))

start_time = time.time()
IL = 1
FL = 3
WL = IL + FL

batch_size = 512

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])


# ==Convolution layer== #
    
with tf.name_scope('Conv1'):
    with tf.name_scope('Input_Decomposition'):
        x_image = tf.py_func(decomposition, [x_image, IL, FL, WL], tf.float32)
    with tf.name_scope('Weights'):
        W_conv1	= tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1)) # 5x5, input_size=1, output_size=32
        W_conv1 = tf.py_func(decomposition, [W_conv1, IL, FL, WL], tf.float32) 
    with tf.name_scope('Biases'):
        b_conv1 = tf.Variable(tf.zeros([32]))     
    with tf.name_scope('Convolution'):
        #h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        h_conv1_arr = []
        h_conv1_result = []
        for i in range(WL):
            h_conv1_arr.append([])
            for j in range(WL):
                h_conv1_arr[i].append(tf.nn.conv2d(x_image[i], W_conv1[j], strides=[1,1,1,1], padding='SAME'))
            sig = IL - 1
            h_conv1_result.append(-h_conv1_arr[i][0]*(2**sig))
            for k in range(1,WL):
                sig -= 1
                h_conv1_result[i] += h_conv1_arr[i][k] * (2**sig)
        sig = IL - 1
        h_conv1 = -h_conv1_result[0]*(2**sig)
        for k in range(1,WL):
            sig -= 1
            h_conv1 += h_conv1_result[k] * (2**sig)
        h_conv1 += b_conv1
    with tf.name_scope('Relu'):
        h_conv1 = tf.nn.relu(h_conv1) # 28x28x32
with tf.name_scope('Maxpooling'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32

with tf.name_scope('Conv2'):
    #with tf.name_scope('Input_Decompddosition'):
        #h_pool1 = tf.py_func(decomposition, [h_pool1, IL, FL, WL], tf.float32) 
    with tf.name_scope('Weights'):
        W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
        #W_conv2 = tf.py_func(decomposition, [W_conv2, IL, FL, WL], tf.float32) 
    with tf.name_scope('Biases'):
        b_conv2 = tf.Variable(tf.zeros(([64])))
    with tf.name_scope('Convolution'):
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
        '''h_conv2_arr = []
        h_conv2_result = []
        for i in range(WL):
            h_conv2_arr.append([])
            for j in range(WL):
                h_conv2_arr[i].append(tf.nn.conv2d(h_pool1[i], W_conv2[j], strides=[1,1,1,1], padding='SAME'))
            sig = IL - 1
            h_conv2_result.append(-h_conv2_arr[i][0]*(2**sig))
            for k in range(1,WL):
                sig -= 1
                h_conv2_result[i] += h_conv2_arr[i][k] * (2**sig)
        sig = IL - 1
        h_conv2 = -h_conv2_result[0]*(2**sig)
        for k in range(1,WL):
            sig -= 1
            h_conv2 += h_conv2_result[k] * (2**sig) 
        h_conv2 += b_conv2'''
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

batch_num = int(mnist.test.num_examples / batch_size)
test_accuracy = 0
print('total batch:', batch_num)
for i in tqdm(range(batch_num)):
    batch = mnist.test.next_batch(batch_size)
    test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
test_accuracy /= batch_num
print('accuracy: %g'%test_accuracy)
print('execution time: %ss' % (time.time() - start_time))
