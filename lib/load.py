import cv2
import numpy as np 
import os 

def load_images(folder):
	# list filenames
	files = os.listdir(folder)

	array_init = False

	for file_name in files:
		image = cv2.imread(folder+file_name, cv2.COLOR_BGR2RGB)
		if array_init:
			data = np.append(data, [image], axis = 0)
		else:
			data = np.array([image])
			array_init = True
	return data

def rgb(folder_name, batch_size, im_width, im_height):
	return np.reshape(load_images(folder_name), [-1, batch_size, im_height, im_width, 3])[:,:,:,:,0]
