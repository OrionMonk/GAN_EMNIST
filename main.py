import tensorflow as tf 
import numpy as np 
from lib import load, generator, shuffle, discriminator
import matplotlib.pyplot as plt
import matplotlib.pyplot as gridspec

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    samples.reshape(-1,23, 19)
    for i in range(samples.shape[0]):
    	ax = plt.subplot(gs[i])
    	plt.axis('off')
    	ax.set_xticklabels([])
    	ax.set_yticklabels([])
    	ax.set_aspect('equal')
    	plt.imshow(samples[i].reshape(23, 19), cmap='Greys_r')

    return fig

def random_noise(size):
	return np.random.uniform(-1., 1., size = size)

def test(folder, im_width, im_height, channels):
	n_epochs = 1000
	batch_size = 100
	random_dim = 100

	# loading 
	data = load.rgb(folder, batch_size, im_width, im_height)
	print(data.shape)

	# shuffle data every epoch
	data = shuffle.shuffle(data, channels, batch_size, im_width, im_height)

	plt.imshow(data[0][0])
	plt.show()

def run(folder, im_width, im_height, channels):
	n_epochs = 10000
	batch_size = 100
	random_dim = 100
	learning_rate = 0.001

	# loading 
	data = load.rgb(folder, batch_size, im_width, im_height)

	# inputs 
	real_images = tf.placeholder(tf.float32, shape = [None, im_height, im_width, channels])
	noise = tf.placeholder(tf.float32, shape = [None, random_dim])

	# computation
	fake_images = generator.generate(noise, random_dim)
	real_prediction = discriminator.discriminate(real_images, im_height, im_width)
	fake_prediction = discriminator.discriminate(fake_images, im_height, im_width, reuse = True)

	# losses
	d_loss = tf.reduce_mean(fake_prediction) - tf.reduce_mean(real_prediction)
	g_loss = -tf.reduce_mean(fake_prediction)

	# variables
	theta_g = [var for var in tf.trainable_variables() if 'gen' in var.name]
	theta_d = [var for var in tf.trainable_variables() if 'dis' in var.name]

	# optimizer
	d_optimize = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = theta_d)
	g_optimize = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = theta_g)


	# training phase
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# initialise variables
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess, 'model_ckpt/pokegan.ckpt')

		for epoch in range(n_epochs):
			# shuffle data every epoch
			data = shuffle.shuffle(data, channels, batch_size, im_width, im_height)

			# train batch
			for batch_id in range(data.shape[0]):
				d_loss_curr, g_loss_curr = sess.run([d_loss, g_loss], feed_dict = {
					real_images:data[batch_id],
					noise: random_noise([batch_size, random_dim])
					})

				if d_loss_curr + 0.5 < g_loss_curr:
					sess.run(g_optimize, feed_dict = {noise:random_noise([batch_size, random_dim])})
				else:
					sess.run(d_optimize, feed_dict = {
						real_images: data[batch_id],
						noise: random_noise([batch_size, random_dim])
						})

			if epoch%10 == 0:
				print('Discriminator Loss: ',d_loss_curr, '\nGenerator Loss: ', g_loss_curr)
				samples = sess.run(fake_images, feed_dict = {noise:random_noise([16, random_dim])})
				fig = plot(samples)
				plt.savefig('out/'+str(epoch)+'.jpg')
				plt.close()

				saver.save(sess, 'model_ckpt/pokegan.ckpt')

if __name__ == "__main__":
	run('Dataset/', 19, 23, 1)
	# test('40x40/', 40, 30, 3)
