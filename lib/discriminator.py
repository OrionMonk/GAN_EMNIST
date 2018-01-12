import tensorflow as tf

def discriminate(x, im_height, im_width, reuse = False):
	with tf.variable_scope('discriminator') as scope:
		if reuse:
			scope.reuse_variables()

		# input layer
		flow = tf.reshape(x, [-1, im_height, im_width, 1])

		# conv layer 1
		flow = tf.layers.conv2d(flow, kernel_size = 7, strides = 1, filters = 48, padding = "VALID", 
			kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		flow = tf.contrib.layers.batch_norm(flow)
		flow = tf.nn.relu(flow)

		# conv layer 2
		flow = tf.layers.conv2d(flow, kernel_size = 7, strides = 1, filters = 96, padding = "VALID", 
			kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False))
		flow = tf.contrib.layers.batch_norm(flow)
		flow = tf.nn.relu(flow)

		# reshape
		flow = tf.reshape(flow, [-1, 11*7*96])

		# fc layer 1
		flow = tf.layers.dense(flow, units = 1)

		return flow