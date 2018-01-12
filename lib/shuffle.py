import numpy as np

def shuffle(data, channels, batch_size, im_width, im_height):
	# reshape array to 2d
	data = np.reshape(data, [-1, im_width * im_height * channels])

	# shuffle data along axis 0
	np.random.shuffle(data)

	return np.reshape(data, [-1, batch_size, im_height, im_width, channels])