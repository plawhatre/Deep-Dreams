import tensorflow as tf
import numpy as np
import copy
from colorama import init
from termcolor import *
import cv2
init()
from io import BytesIO
import numpy as np
import requests
import matplotlib.pyplot as plt

def preprocess(I):
	return 2*I/255 - 1

def deprocess(I):
	return 255*(I + 1)/2

class DeepDreamModel(tf.keras.layers.Layer):
	def __init__(self, model, X):
		super(DeepDreamModel, self).__init__()
		self.model = model
		# self.X = tf.Variable(X, name='GenImg', trainable=True)
		self.X = X

	def forward(self, X):
		Y_dream = self.model(X) 	
		return Y_dream

	def clip(self, X):
	  return tf.clip_by_value(X, clip_value_min=-1.0, clip_value_max=1.0)

	def loss(self, X):
		a_mean = []
		for activation in X:
			a_mean.append(tf.math.reduce_mean(activation))

		return tf.math.reduce_sum(a_mean)

	def octave_train(self, J, n, lr):
		X = tf.Variable(np.expand_dims(J, axis=0))
		optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
		for epoch in range(35):
			with tf.GradientTape() as t:
				t.watch(X)
				Y = self.forward(X)
				Loss = self.loss(Y)

			grads = t.gradient(Loss, X)
			# Normalizing gradients
			grads /= tf.math.reduce_std(grads) + 1e-8
			# Gradient Ascent
			grads *= -1
			cprint(f'Octave scale: {n}, Epoch: {epoch}, Loss: {Loss}','green')	

			optimizer.apply_gradients(zip([grads], [X]))
			X.assign(self.clip(X))

		return np.squeeze(X.numpy(), axis=0)

	def train(self, lr=0.01):
		int_input_shape = tf.shape(self.X)[:-1]
		float_input_shape = tf.cast(int_input_shape, tf.float32)
		for n in range(-2, 3):
			new_shape = tf.cast(float_input_shape*1.5**n, tf.int32)
			self.X = tf.image.resize(self.X, new_shape)
			self.X = self.octave_train(self.X, n, lr)

		self.X = tf.image.resize(self.X, int_input_shape)

		return self.X


if __name__ == '__main__':
	# Content Image
	# C = cv2.imread('Taj-Mahal.jpg')
	C = cv2.imread('Content/Taj-Mahal.jpg')
	# C = cv2.resize(C, (224, 224))
	C = preprocess(C)

	# Initiliatize Model
	base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	# Maximize the activations of these layers
	names = ['mixed3', 'mixed5']
	layers = [base_model.get_layer(name).output for name in names]

	# Create the feature extraction model
	dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

	X = tf.Variable(tf.cast(C, tf.float32), name='GenImg', trainable=True)
	model = DeepDreamModel(dream_model, X)
	X = model.train()
	X = deprocess(X)


	# I = np.squeeze(X.numpy(), axis=0)
	I = X.numpy()
	# I = X.numpy().reshape(224,224,3)
	plt.figure()
	plt.imshow(I / 255)
	plt.title('DeepDream')
	plt.show()
	cv2.imwrite('DeepDream.jpg', I)
