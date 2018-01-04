import tensorflow as tf

x = tf.constant([[[[1],[2]],[[3],[4]]]], tf.float32) #NHWC
filt = tf.constant([[[[1,1]],[[2,2]]], [[[3,3]], [[4,4]]]], tf.float32) #HW, inC, outC

x0 = tf.mod(x, 2)
x1 = tf.floordiv(x, 2)
result = tf.nn.conv2d(x, filt, strides=[1,1,1,1], padding='VALID')
#w = tf.Variable(tf.truncated_normal([2,2,1,2], stddev=0.1))

#x_modi = tf.concat([x0, x1], 0)
#result_modi = tf.nn.conv2d(x_modi, filt, strides=[1,1,1,1], padding='VALID')

result0 = tf.nn.conv2d(x0, filt, strides=[1,1,1,1], padding='VALID')
result1 = tf.nn.conv2d(x1, filt, strides=[1,1,1,1], padding='VALID')
result1 = tf.multiply(result1, 2)
result_modi = tf.add(result0, result1)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
print('x0',sess.run(x0))
print('x1',sess.run(x1))
print('x.shape',sess.run(tf.shape(x)))
#print('x_modi.shape',sess.run(tf.shape(x_modi)))
print('filter.shape',sess.run(tf.shape(filt)))
#print(sess.run(tf.shape(w)))
print('result',sess.run(result))
print('result_modi',sess.run(result_modi))

# Visualize
writer = tf.summary.FileWriter('TensorBoard/', graph=sess.graph)
sess.close()

