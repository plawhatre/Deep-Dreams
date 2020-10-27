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
	return 2.0*I/255 - 1

def deprocess(I):
	return 255*(I + 1)/2.0

class DeepDreamModel(tf.keras.layers.Layer):
	def __init__(self, model, X, C, S):
		super(DeepDreamModel, self).__init__()
		self.model = model
		self.X = X
		self.C = C
		self.S = S

	def forward(self, X):
		Y_dream = self.model(X) 	
		return Y_dream

	def clip(self, X):
	  return tf.clip_by_value(X, clip_value_min=-1.0, clip_value_max=1.0)

	def content_loss(self, X, C):
		for act_x, act_c in zip([X[5]], [C[5]]):
			out = 0.5*tf.norm(act_x - act_c, ord=2)**2

		return out

	def style_loss(self, weight_s, X, S):
		out = 0
		for act_x, act_s, ws in zip([X[0],X[1],X[2],X[3],X[4]], [S[0],S[1],S[2],S[3],S[4]],weight_s):
			_, x_h, x_w, _ = tf.shape(act_x)
			G_x = tf.linalg.einsum('bijc, bijd->bcd', act_x, act_x)/ (tf.cast(x_h, tf.float32)*tf.cast(x_w, tf.float32))
			
			_, s_h, s_w, _ = tf.shape(act_s)
			G_s = tf.linalg.einsum('bijc, bijd->bcd', act_s, act_s)/ (tf.cast(s_h, tf.float32)*tf.cast(s_w, tf.float32))

			out += ws * 0.25 * tf.norm(G_x-G_s, ord='fro', axis=(1,2))**2

		return out

	def DeepDream_loss(self, X, gamma):
		a_mean = []
		for activation in [X[1],X[5]]:
			a_mean.append(tf.math.reduce_mean(gamma * activation))

		return tf.math.reduce_sum(a_mean)

	def NST_loss(self, X, C, S, weight_s, alpha, beta):
		out1 = alpha*self.content_loss(X, C)
		out2 = beta*self.style_loss(weight_s, X, S)
		out = out1 + out2

		return out

	def octave_train(self, J, n, weight_s, alpha, beta, gamma, lr):
		X = tf.Variable(np.expand_dims(J, axis=0))
		C = np.expand_dims(self.C, axis=0)
		S = np.expand_dims(self.S, axis=0)
		nst_optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
		dd_optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
		for epoch in range(2000):
			with tf.GradientTape(persistent=True) as t:
				t.watch(X)
				Y_x = self.forward(X)
				Y_c = self.forward(C)
				Y_s = self.forward(S)
				NST_Loss = self.NST_loss(Y_x, Y_c, Y_s, weight_s, alpha, beta)
				DeepDream_Loss = self.DeepDream_loss(Y_x, gamma)

			nst_grads = t.gradient(NST_Loss, X)
			dd_grads = t.gradient(DeepDream_Loss, X)
			# Normalizing gradients
			nst_grads /= tf.math.reduce_std(nst_grads) + 1e-8
			dd_grads /= tf.math.reduce_std(dd_grads) + 1e-8
			dd_grads *= -1

			if epoch % 10 ==0:
				cprint(f'Octave scale: {n}, Epoch: {epoch}, NST_Loss: {NST_Loss}, DD_Loss: {DeepDream_Loss}','green')	

			nst_optimizer.apply_gradients(zip([nst_grads], [X]))
			dd_optimizer.apply_gradients(zip([dd_grads], [X]))
			X.assign(self.clip(X))

		return np.squeeze(X.numpy(), axis=0)

	def train(self, weight_s, alpha, beta, gamma, lr=0.01):
		int_input_shape = tf.shape(self.X)[:-1]
		float_input_shape = tf.cast(int_input_shape, tf.float32)
		# for n in range(-1, 2):
		for n in [0]:
			new_shape = tf.cast(float_input_shape*1.5**n, tf.int32)
			self.X = tf.image.resize(self.X, new_shape)
			self.C = tf.image.resize(self.C, new_shape)
			self.S = tf.image.resize(self.S, new_shape)
			self.X = self.octave_train(self.X, n, weight_s, alpha, beta, gamma, lr)

		self.X = tf.image.resize(self.X, int_input_shape)

		return self.X


if __name__ == '__main__':
	weight_s=[0.25/64, 0.25/128, 0.25/256, 0.25/512, 0.25/512]
	alpha = 1e2
	beta = 5e4 
	gamma = 5e3
	# Content Image
	C = cv2.imread('Taj-Mahal.jpg')
	# C = cv2.imread('Content/Taj-Mahal.jpg')
	C = preprocess(C)

	# Style Image
	S = cv2.imread('starry_night.jpg')
	# S = cv2.imread('Style/starry_night.jpg')
	S = preprocess(S)

	# Initiliatize Model
	base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
	# Maximize the activations of these layers
	names = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1',
                'block4_conv2']
	layers = [base_model.get_layer(name).output for name in names]

	# Create the feature extraction model
	dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

	X = tf.Variable(np.random.random((C.shape[0],C.shape[1],C.shape[2])), name='GenImg', trainable=True)
	# X = tf.Variable(C, name='GenImg', trainable=True)
	model = DeepDreamModel(dream_model, X, C, S)
	X = model.train(weight_s, alpha, beta, gamma)
	
	X = deprocess(X)
	C = deprocess(C)
	S = deprocess(S)


	# I = np.squeeze(X.numpy(), axis=0)
	I = X.numpy()
	# I = X.numpy().reshape(224,224,3)
	plt.figure()
	plt.imshow(I / 255)
	plt.title('DeepDream_NST')
	plt.show()
	cv2.imwrite('DeepDream_NST.jpg', I)
