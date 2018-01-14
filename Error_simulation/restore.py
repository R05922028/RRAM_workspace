import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Testing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Restore Model
sess = tf.Session()
saver = tf.train.import_meta_graph('./model/cnn_model.ckpt.meta')
saver.restore(sess, './model/cnn_model.ckpt')
graph = tf.get_default_graph()

x = graph.get_tensor_by_name('x:0')
y_ = graph.get_tensor_by_name('y_:0')
prediction = graph.get_tensor_by_name('prediction:0')

# Testing
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)  
