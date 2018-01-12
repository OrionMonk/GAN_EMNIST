# Generative Adversarial Networks for generating Extended-MNIST samples
### Training
One fully connected layer and two convolution layers for both the generator and the discriminator were used. The neural network was trained for *approximately* 2k epochs, with training samples of the *extended-MNIST dataset*, which contains handwritten digits as well as letters of the English Alphabet. 

The performance of the network seemed to be very sensitive to the hyper-parameters and initializations, especially the learning rate. Too large a learning rate caused the GAN to diverge from the minimum, whereas too small almost always ultimately led Mode Collapse.

The use of xavier initializer and relu layer (non leaky) gave best results.

### Generated Samples
<img src="https://github.com/OrionMonk/GAN_EMNIST/blob/master/train.gif" width="50%">

*Fig. Samples generated during training*
