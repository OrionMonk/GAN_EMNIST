import tensorflow as tf

def generate(y, random_dim):
	with tf.variable_scope('generator') as scope:
		# input layer
		flow = tf.reshape(y, [-1, random_dim])

		# fc layer 1
		flow = tf.layers.dense(flow, units = 11*7*96, kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		flow = tf.contrib.layers.batch_norm(flow)
		flow = tf.nn.relu(flow)

		# reshaping
		flow = tf.reshape(flow, [-1, 11, 7, 96])

		# deconv layer 1
		flow = tf.layers.conv2d_transpose(flow, strides = 1, kernel_size = 7, padding = "VALID", filters = 48, 
			kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		flow = tf.contrib.layers.batch_norm(flow)
		flow = tf.nn.relu(flow)

		# deconv layer 1
		flow = tf.layers.conv2d_transpose(flow, strides = 1, kernel_size = 7, padding = "VALID", filters = 1, 
			kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		flow = tf.contrib.layers.batch_norm(flow)
		flow = tf.nn.tanh(flow)

		return flow
