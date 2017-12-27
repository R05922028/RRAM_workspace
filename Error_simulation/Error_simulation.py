import tensorflow as tf
import numpy as np

input_data = np.array([[0,2,1,3,1,2,2,0,1],[0,2,1,3,1,2,2,0,1]])
input_data = input_data.reshape(-1,3,3,1)
#filters = np.array([[0,2,1,3]])
filters = np.array([[1,0,1,0],[1,1,1,1]])
filters = filters.reshape(2,2,1,-1)
print(input_data)
print(filters)

# Convert to tensor
input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
filters = tf.convert_to_tensor(filters, dtype=tf.float32)
 
#a = tf.mod(input_data, 2)
#b = tf.floordiv(input_data, 2)
#input_data = tf.concat([a,b],0)

#a = tf.mod(filters, 2)
#b = tf.floordiv(filters, 2)
#filters = tf.concat([a,b], 0)
#filters = tf.reshape(filters, [2,2,1,2])


# Convolution 2D
result = tf.nn.conv2d(input_data, filters, strides=[1,1,1,1], padding='VALID', data_format='NHWC')

sess = tf.InteractiveSession()
#print('input\n',sess.run(input_data))
#print('filters shape:',sess.run(tf.shape(filters)))
#print('filters\n',sess.run(filters))

print('result shape:',sess.run(tf.shape(result)))
print('result\n',sess.run(result))
