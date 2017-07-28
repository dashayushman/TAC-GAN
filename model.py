import tensorflow as tf
import tensorflow.contrib.slim as slim
from Utils import ops


class GAN :
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''

	def __init__(self, options) :
		self.options = options

	def build_model(self) :

		print('Initializing placeholder')
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],
		                              img_size, img_size, 3],
		                              name = 'real_image')
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],
                                       img_size, img_size, 3],
		                               name = 'wrong_image')

		t_real_caption = tf.placeholder('float32', [self.options['batch_size'],
				                        self.options['caption_vector_length']],
		                                name='real_captions')

		t_z = tf.placeholder('float32', [self.options['batch_size'],
		                      self.options['z_dim']], name='input_noise')

		t_real_classes = tf.placeholder('float32', [self.options['batch_size'],
										self.options['n_classes']],
		                                name='real_classes')

		t_wrong_classes = tf.placeholder('float32', [self.options['batch_size'],
										 self.options['n_classes']],
		                                 name='wrong_classes')

		t_training = tf.placeholder(tf.bool, name='training')

		print('Building the Generator')
		fake_image = self.generator(t_z, t_real_caption,
												 t_training)

		print('Building the Discriminator')
		disc_real_image, disc_real_image_logits, disc_real_image_aux, \
			disc_real_image_aux_logits = self.discriminator(
				t_real_image, t_real_caption, self.options['n_classes'],
				t_training)

		disc_wrong_image, disc_wrong_image_logits, disc_wrong_image_aux, \
			disc_wrong_image_aux_logits  = self.discriminator(
				t_wrong_image, t_real_caption, self.options['n_classes'],
				t_training, reuse = True)

		disc_fake_image, disc_fake_image_logits, disc_fake_image_aux, \
			disc_fake_image_aux_logits  = self.discriminator(
				fake_image, t_real_caption, self.options['n_classes'],
				t_training, reuse = True)

		d_right_predictions = tf.equal(tf.argmax(disc_real_image_aux, 1),
		                               tf.argmax(t_real_classes, 1))
		d_right_accuracy = tf.reduce_mean(tf.cast(d_right_predictions,
		                                          tf.float32))

		d_wrong_predictions = tf.equal(tf.argmax(disc_wrong_image_aux, 1),
		                               tf.argmax(t_wrong_classes, 1))
		d_wrong_accuracy = tf.reduce_mean(tf.cast(d_wrong_predictions,
		                                          tf.float32))

		d_fake_predictions = tf.equal(tf.argmax(disc_fake_image_aux_logits, 1),
		                              tf.argmax(t_real_classes, 1))
		d_fake_accuracy = tf.reduce_mean(tf.cast(d_fake_predictions,
		                                         tf.float32))

		tf.get_variable_scope()._reuse = False

		print('Building the Loss Function')
		g_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
								  logits=disc_fake_image_logits,
								  labels=tf.ones_like(disc_fake_image)))

		g_loss_2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
											logits=disc_fake_image_aux_logits,
	                                        labels=t_real_classes))

		d_loss1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
										logits=disc_real_image_logits,
			                            labels=tf.ones_like(disc_real_image)))
		d_loss1_1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
											logits=disc_real_image_aux_logits,
	                                        labels=t_real_classes))
		d_loss2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
									logits=disc_wrong_image_logits,
                                    labels=tf.zeros_like(disc_wrong_image)))
		d_loss2_1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
											logits=disc_wrong_image_aux_logits,
	                                        labels=t_wrong_classes))
		d_loss3 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
										logits=disc_fake_image_logits,
										labels=tf.zeros_like(disc_fake_image)))

		d_loss = d_loss1 + d_loss1_1 + d_loss2 + d_loss2_1 + d_loss3 + g_loss_2
		
		g_loss = g_loss_1 + g_loss_2

		t_vars = tf.trainable_variables()
		print('List of all variables')
		for v in t_vars:
			print(v.name)
			print(v)
			self.add_histogram_summary(v.name, v)

		self.add_tb_scalar_summaries(d_loss, g_loss, d_loss1, d_loss2, d_loss3,
              d_loss1_1, d_loss2_1, g_loss_1, g_loss_2, d_right_accuracy,
              d_wrong_accuracy, d_fake_accuracy)

		self.add_image_summary('Generated Images', fake_image,
		                       self.options['batch_size'])

		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z,
			't_real_classes' : t_real_classes,
			't_wrong_classes' : t_wrong_classes,
			't_training' : t_training,

		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			'd_loss1': d_loss1,
			'd_loss2': d_loss2,
			'd_loss3': d_loss3,
			'g_loss_1': g_loss_1,
			'g_loss_2': g_loss_2,
			'd_loss1_1': d_loss1_1,
			'd_loss2_1': d_loss2_1,
			'disc_real_image_logits': disc_real_image_logits,
			'disc_wrong_image_logits': disc_wrong_image,
			'disc_fake_image_logits': disc_fake_image_logits
		}

		return input_tensors, variables, loss, outputs, checks

	def add_tb_scalar_summaries(self, d_loss, g_loss, d_loss1, d_loss2,
	                              d_loss3, d_loss1_1, d_loss2_1, g_loss_1,
	                              g_loss_2, d_right_accuracy,
	                              d_wrong_accuracy, d_fake_accuracy):

		self.add_scalar_summary("D_Loss", d_loss)
		self.add_scalar_summary("G_Loss", g_loss)
		self.add_scalar_summary("D loss-1 [Real/Fake loss for real images]",
		                        d_loss1)
		self.add_scalar_summary("D loss-2 [Real/Fake loss for wrong images]",
		                        d_loss2)
		self.add_scalar_summary("D loss-3 [Real/Fake loss for fake images]",
		                        d_loss3)
		self.add_scalar_summary(
			"D loss-4 [Aux Classifier loss for real images]", d_loss1_1)
		self.add_scalar_summary(
			"D loss-5 [Aux Classifier loss for wrong images]", d_loss2_1)
		self.add_scalar_summary("G loss-1 [Real/Fake loss for fake images]",
		                        g_loss_1)
		self.add_scalar_summary(
			"G loss-2 [Aux Classifier loss for fake images]", g_loss_2)
		self.add_scalar_summary("Discriminator Real Image Accuracy",
		                        d_right_accuracy)
		self.add_scalar_summary("Discriminator Wrong Image Accuracy",
		                        d_wrong_accuracy)
		self.add_scalar_summary("Discriminator Fake Image Accuracy",
		                        d_fake_accuracy)

	def add_scalar_summary(self, name, var):
		with tf.name_scope('summaries'):
			tf.summary.scalar(name, var)

	def add_histogram_summary(self, name, var):
		with tf.name_scope('summaries'):
			tf.summary.histogram(name, var)

	def add_image_summary(self, name, var, max_outputs=1):
		with tf.name_scope('summaries'):
			tf.summary.image(name, var, max_outputs=max_outputs)

	# GENERATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding, t_training):

		s = self.options['image_size']
		s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

		reduced_text_embedding = ops.lrelu(
			ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
		z_concat = tf.concat([t_z, reduced_text_embedding], -1)
		z_ = ops.linear(z_concat, self.options['gf_dim'] * 8 * s16 * s16,
		                'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(slim.batch_norm(h0, is_training = t_training,
		                                scope="g_bn0"))

		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8,
		                       self.options['gf_dim'] * 4], name = 'g_h1')
		h1 = tf.nn.relu(slim.batch_norm(h1, is_training = t_training,
		                                scope="g_bn1"))

		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4,
		                       self.options['gf_dim'] * 2], name = 'g_h2')
		h2 = tf.nn.relu(slim.batch_norm(h2, is_training = t_training,
		                                scope="g_bn2"))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2,
		                       self.options['gf_dim'] * 1], name = 'g_h3')
		h3 = tf.nn.relu(slim.batch_norm(h3, is_training = t_training,
		                                scope="g_bn3"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3],
		                  name = 'g_h4')
		return (tf.tanh(h4) / 2. + 0.5)


	# DISCRIMINATOR IMPLEMENTATION based on :
	# https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, n_classes, t_training,
	                  reuse = False) :
		if reuse :
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(
			ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv'))  # 64

		h1 = ops.lrelu(slim.batch_norm(ops.conv2d(h0,
		                                     self.options['df_dim'] * 2,
		                                     name = 'd_h1_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn1'))  # 32

		h2 = ops.lrelu(slim.batch_norm(ops.conv2d(h1,
		                                     self.options['df_dim'] * 4,
		                                     name = 'd_h2_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn2'))  # 16
		h3 = ops.lrelu(slim.batch_norm(ops.conv2d(h2,
		                                     self.options['df_dim'] * 8,
		                                     name = 'd_h3_conv'),
		                               reuse=reuse,
		                               is_training = t_training,
		                               scope = 'd_bn3'))  # 8
		h3_shape = h3.get_shape().as_list()
		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding,
		                                               self.options['t_dim'],
		                                               'd_embedding'))
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
		reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)
		tiled_embeddings = tf.tile(reduced_text_embeddings,
		                           [1, h3_shape[1], h3_shape[1], 1],
		                           name = 'tiled_embeddings')

		h3_concat = tf.concat([h3, tiled_embeddings], 3, name = 'h3_concat')
		h3_new = ops.lrelu(slim.batch_norm(ops.conv2d(h3_concat,
												self.options['df_dim'] * 8,
													  1, 1, 1, 1,
												name = 'd_h3_conv_new'),
		                                reuse=reuse,
		                                is_training = t_training,
		                                scope = 'd_bn4'))  # 4

		h3_flat = tf.reshape(h3_new, [self.options['batch_size'], -1])

		h4 = ops.linear(h3_flat, 1, 'd_h4_lin_rw')
		h4_aux = ops.linear(h3_flat, n_classes, 'd_h4_lin_ac')
		
		return tf.nn.sigmoid(h4), h4, tf.nn.sigmoid(h4_aux), h4_aux

	# This has not been used used yet but can be used
	def attention(self, decoder_output, seq_outputs, output_size, time_steps,
			reuse=False) :
		if reuse:
			tf.get_variable_scope().reuse_variables()
		ui = ops.attention(decoder_output, seq_outputs, output_size,
		                   time_steps, name = "g_a_attention")

		with tf.variable_scope('g_a_attention'):
			ui = tf.transpose(ui, [1, 0, 2])
			ai = tf.nn.softmax(ui,  dim=1)
			seq_outputs = tf.transpose(seq_outputs, [1, 0, 2])
			d_dash = tf.reduce_sum(tf.mul(seq_outputs, ai), axis=1)
			return d_dash, ai
