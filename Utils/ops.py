# RESUED CODE FROM https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


class batch_norm(object):
	"""Code modification of http://stackoverflow.com/a/33950177"""
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum

			self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
			self.name = name

	def __call__(self, x, train=True):
		shape = x.get_shape().as_list()

		if train:
			with tf.variable_scope(self.name) as scope:
				self.beta = tf.get_variable("beta", [shape[-1]],
									initializer=tf.constant_initializer(0.))
				self.gamma = tf.get_variable("gamma", [shape[-1]],
									initializer=tf.random_normal_initializer(1., 0.02))
				
				try:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
				except:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
					
				ema_apply_op = self.ema.apply([batch_mean, batch_var])
				self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

				with tf.control_dependencies([ema_apply_op]):
					mean, var = tf.identity(batch_mean), tf.identity(batch_var)
		else:
			mean, var = self.ema_mean, self.ema_var

		normed = tf.nn.batch_norm_with_global_normalization(
				x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

		return normed



def binary_cross_entropy(preds, targets, name=None):
	"""Computes binary cross entropy given `preds`.

	For brevity, let `x = `, `z = targets`.  The logistic loss is

		loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

	Args:
		preds: A `Tensor` of type `float32` or `float64`.
		targets: A `Tensor` of the same type and shape as `preds`.
	"""
	eps = 1e-12
	with ops.op_scope([preds, targets], name, "bce_loss") as name:
		preds = ops.convert_to_tensor(preds, name="preds")
		targets = ops.convert_to_tensor(targets, name="targets")
		return tf.reduce_mean(-(targets * tf.log(preds + eps) +
							  (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		#w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer = tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		#w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]], initializer = tf.contrib.layers.xavier_initializer())
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0,
           with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		#matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
		#						 tf.random_normal_initializer(stddev=stddev))
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

def attention(decoder_output, seq_outputs, output_size, time_steps,
              name="attention"):

	with tf.variable_scope(name):
		ui = []
		w_1 = tf.get_variable("w1", [output_size, output_size],
							  tf.float32,
							  tf.contrib.layers.xavier_initializer())
		w_2 = tf.get_variable("w2", [output_size, output_size],
							  tf.float32,
							  tf.contrib.layers.xavier_initializer())
		v = tf.get_variable("v", [output_size, 1],
							  tf.float32,
							  tf.contrib.layers.xavier_initializer())
		for seq_out in seq_outputs:
			ui.append(tf.matmul(tf.nn.tanh(tf.matmul(seq_out, w_1) +
									tf.matmul(decoder_output, w_2)), v))

		return ui

def get_gt(batch_size, classes, real=1, name="gt"):

	with tf.variable_scope(name, reuse=None):
		r_f = tf.get_variable("rf", [batch_size, 1],
		                       initializer = tf.constant_initializer(
			                       real))
		gt = tf.concat(1, [r_f, classes], name = 'gt_concat_classes')
		return gt