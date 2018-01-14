from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

if not os.path.exists('./model'):
    os.makedirs('./model')

epoch = 50
DAC_bit_width = 1

def preprocessing(x):
    return x
def error_injection(x):
    return x

# 55,000 training data
# 10,000 testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784], name='x') / 255.
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ==Convolution layer== #
with tf.name_scope('Conv1'):
    W_conv1	= tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1), name='weight') # 5x5, input_size=1, output_size=32
    b_conv1 = tf.Variable(tf.zeros([32]), name='bias')
    with tf.name_scope('convolution'):
        h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME', name='Convolution')
        h_conv1 = tf.add(h_conv1, b_conv1, name='Add')
    h_conv1 = tf.nn.relu(h_conv1, name='Relu') # 28x28x32
with tf.name_scope('Maxpooling'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14x14x32
with tf.name_scope('Conv2'):
    W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='weight')
    b_conv2 = tf.Variable(tf.zeros(([64])), name='bias')
    with tf.name_scope('data_decomposition'):
        h_pool1 = tf.py_func(preprocessing, [h_pool1], tf.float32)
    with tf.name_scope('convolution'):
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME', name='Convolution')
        h_conv2 = tf.add(h_conv2, b_conv2, name='Add')
    with tf.name_scope('error_injection'):
        h_conv2 = tf.py_func(error_injection, [h_conv2], tf.float32)
    h_conv2 = tf.nn.relu(h_conv2, name='Relu') # 14x14x64
with tf.name_scope('Maxpooling'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 7x7x64
# ==Fully connected layer== #
with tf.name_scope('Fully-connected'):
#with tf.name_scope('Dense1'):
    W_fcon1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='weight')
    b_fcon1 = tf.Variable(tf.zeros([1024]), name='bias')
    flatten = tf.reshape(h_pool2, [-1,7*7*64], name='Flatten')
    h_fcon1 = tf.matmul(flatten, W_fcon1, name='Multiply')
    h_fcon1 = tf.add(h_fcon1, b_fcon1, name='Add')
    h_fcon1 = tf.nn.relu(h_fcon1, name='Relu')
    h_drop1 = tf.nn.dropout(h_fcon1, 0.5, name='Dropout')
#with tf.name_scope('Dense2'):
    W_fcon2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='Weight')
    b_fcon2 = tf.Variable(tf.zeros([10]), name='bias')
    h_fcon2 = tf.matmul(h_drop1, W_fcon2, name='Multiply')
    h_fcon2 = tf.add(h_fcon2, b_fcon2, name='Add')

with tf.name_scope('Output'):
    prediction = tf.identity(h_fcon2, name='prediction')

with tf.name_scope('Train'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# initialize Graph
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)


# Training
for step in range(epoch):
    batch = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x: batch[0], y_:batch[1]})
    if step %20 == 0:
        print('Loss:', sess.run(loss, feed_dict={x: batch[0], y_: batch[1]}))
'''
# Save model
saver = tf.train.Saver()
save_path = saver.save(sess, "./model/cnn_model.ckpt")
print("Model saved in file: %s" % save_path)
'''
# Testing
ans = tf.placeholder(tf.float32, shape=[None, 10], name='ans')
pre = sess.run(prediction, feed_dict={x: mnist.test.images})
correct_prediction = tf.equal(tf.argmax(pre,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("accuracy: ", sess.run(accuracy, feed_dict={ans:mnist.test.labels}))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()
