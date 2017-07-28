import model
import argparse
import pickle
import scipy.misc
import random
import os
import shutil

import tensorflow as tf
import numpy as np

from os.path import join
from Utils import image_processing

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
						help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default=256,
						help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=64,
						help='Batch Size')

	parser.add_argument('--image_size', type=int, default=128,
						help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64,
						help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64,
						help='Number of conv in the first layer discr.')

	parser.add_argument('--caption_vector_length', type=int, default=4800,
						help='Caption Vector Length')

	parser.add_argument('--n_classes', type = int, default = 102,
	                    help = 'Number of classes/class labels')

	parser.add_argument('--data_dir', type=str, default="Data",
						help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
						help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
						help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=200,
						help='Max number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
						help='Save Model/Samples every x iterations over '
							 'batches')

	parser.add_argument('--resume_model', type=bool, default=False,
						help='Pre-Trained Model load or not')

	parser.add_argument('--data_set', type=str, default="flowers",
						help='Dat set: MS-COCO, flowers')

	parser.add_argument('--model_name', type=str, default="TAC_GAN",
						help='model_1 or model_2')

	parser.add_argument('--train', type = bool, default = True,
	                    help = 'True while training and otherwise')

	args = parser.parse_args()

	model_dir, model_chkpnts_dir, model_samples_dir, model_val_samples_dir,\
							model_summaries_dir = initialize_directories(args)

	datasets_root_dir = join(args.data_dir, 'datasets')
	loaded_data = load_training_data(datasets_root_dir, args.data_set,
	                                 args.caption_vector_length,
	                                 args.n_classes)
	model_options = {
		'z_dim': args.z_dim,
		't_dim': args.t_dim,
		'batch_size': args.batch_size,
		'image_size': args.image_size,
		'gf_dim': args.gf_dim,
		'df_dim': args.df_dim,
		'caption_vector_length': args.caption_vector_length,
		'n_classes': loaded_data['n_classes']
	}

	# Initialize and build the GAN model
	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()

	d_optim = tf.train.AdamOptimizer(args.learning_rate,
									 beta1=args.beta1).minimize(loss['d_loss'],
											var_list=variables['d_vars'])
	g_optim = tf.train.AdamOptimizer(args.learning_rate,
									 beta1=args.beta1).minimize(loss['g_loss'],
											var_list=variables['g_vars'])

	global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
	merged = tf.summary.merge_all()
	sess = tf.InteractiveSession()

	summary_writer = tf.summary.FileWriter(model_summaries_dir, sess.graph)

	tf.global_variables_initializer().run()
	saver = tf.train.Saver(max_to_keep=10000)

	if args.resume_model:
		print('Trying to resume training from a previous checkpoint' +
		      str(tf.train.latest_checkpoint(model_chkpnts_dir)))
		if tf.train.latest_checkpoint(model_chkpnts_dir) is not None:
			saver.restore(sess, tf.train.latest_checkpoint(model_chkpnts_dir))
			print('Successfully loaded model. Resuming training.')
		else:
			print('Could not load checkpoints.  Training a new model')
	global_step = global_step_tensor.eval()
	gs_assign_op = global_step_tensor.assign(global_step)
	for i in range(args.epochs):
		batch_no = 0
		while batch_no * args.batch_size + args.batch_size < \
				loaded_data['data_length']:

			real_images, wrong_images, caption_vectors, z_noise, image_files, \
			real_classes, wrong_classes, image_caps, image_ids = \
							   get_training_batch(batch_no, args.batch_size,
	                                              args.image_size, args.z_dim,
	                                              'train', datasets_root_dir,
	                                              args.data_set, loaded_data)

			# DISCR UPDATE
			check_ts = [checks['d_loss1'], checks['d_loss2'],
			            checks['d_loss3'], checks['d_loss1_1'],
			            checks['d_loss2_1']]

			feed = {
				input_tensors['t_real_image'].name : real_images,
				input_tensors['t_wrong_image'].name : wrong_images,
				input_tensors['t_real_caption'].name : caption_vectors,
				input_tensors['t_z'].name : z_noise,
				input_tensors['t_real_classes'].name : real_classes,
				input_tensors['t_wrong_classes'].name : wrong_classes,
				input_tensors['t_training'].name : args.train
			}

			_, d_loss, gen, d1, d2, d3, d4, d5= sess.run([d_optim,
                        loss['d_loss'],outputs['generator']] + check_ts,
                        feed_dict=feed)

			print("D total loss: {}\n"
			      "D loss-1 [Real/Fake loss for real images] : {} \n"
			      "D loss-2 [Real/Fake loss for wrong images]: {} \n"
			      "D loss-3 [Real/Fake loss for fake images]: {} \n"
			      "D loss-4 [Aux Classifier loss for real images]: {} \n"
			      "D loss-5 [Aux Classifier loss for wrong images]: {}"
			      " ".format(d_loss, d1, d2, d3, d4, d5))

			# GEN UPDATE
			_, g_loss, gen = sess.run([g_optim, loss['g_loss'],
                                       outputs['generator']], feed_dict=feed)

			# GEN UPDATE TWICE
			_, summary, g_loss, gen, g1, g2 = sess.run([g_optim, merged,
                   loss['g_loss'], outputs['generator'], checks['g_loss_1'],
                   checks['g_loss_2']], feed_dict=feed)
			summary_writer.add_summary(summary, global_step)
			print("\n\nLOSSES\nDiscriminator Loss: {}\nGenerator Loss: {"
                  "}\nBatch Number: {}\nEpoch: {},\nTotal Batches per "
                  "epoch: {}\n".format( d_loss, g_loss, batch_no, i,
                    int(len(loaded_data['image_list']) / args.batch_size)))
			print("\nG loss-1 [Real/Fake loss for fake images] : {} \n"
			      "G loss-2 [Aux Classifier loss for fake images]: {} \n"
			      " ".format(g1, g2))
			global_step += 1
			sess.run(gs_assign_op)
			batch_no += 1
			if (batch_no % args.save_every) == 0 and batch_no != 0:
				print("Saving Images and the Model\n\n")

				save_for_vis(model_samples_dir, real_images, gen, image_files,
				             image_caps, image_ids)
				save_path = saver.save(sess,
                                       join(model_chkpnts_dir,
				                            "latest_model_{}_temp.ckpt".format(
										        args.data_set)))

				# Getting a batch for validation
				val_captions, val_image_files, val_image_caps, val_image_ids = \
                          get_val_caps_batch(args.batch_size, loaded_data,
                                             args.data_set, datasets_root_dir)

				shutil.rmtree(model_val_samples_dir)
				os.makedirs(model_val_samples_dir)

				for val_viz_cnt in range(0, 4):
					val_z_noise = np.random.uniform(-1, 1, [args.batch_size,
					                                        args.z_dim])

					val_feed = {
						input_tensors['t_real_caption'].name : val_captions,
						input_tensors['t_z'].name : val_z_noise,
						input_tensors['t_training'].name : True
					}

					val_gen = sess.run([outputs['generator']],
					                   feed_dict=val_feed)
					save_for_viz_val(model_val_samples_dir, val_gen,
					                 val_image_files, val_image_caps,
									 val_image_ids, args.image_size,
									 val_viz_cnt)

		# Save the model after every epoch
		if i % 1 == 0:
			epoch_dir = join(model_chkpnts_dir, str(i))
			if not os.path.exists(epoch_dir):
				os.makedirs(epoch_dir)

			save_path = saver.save(sess,
			                       join(epoch_dir,
			                            "model_after_{}_epoch_{}.ckpt".
			                                format(args.data_set, i)))
			val_captions, val_image_files, val_image_caps, val_image_ids = \
				  get_val_caps_batch(args.batch_size, loaded_data,
				                     args.data_set, datasets_root_dir)

			shutil.rmtree(model_val_samples_dir)
			os.makedirs(model_val_samples_dir)

			for val_viz_cnt in range(0, 10):
				val_z_noise = np.random.uniform(-1, 1, [args.batch_size,
				                                        args.z_dim])
				val_feed = {
					input_tensors['t_real_caption'].name : val_captions,
					input_tensors['t_z'].name : val_z_noise,
					input_tensors['t_training'].name : True
				}
				val_gen = sess.run([outputs['generator']], feed_dict=val_feed)
				save_for_viz_val(model_val_samples_dir, val_gen,
				                 val_image_files, val_image_caps,
								 val_image_ids, args.image_size,
								 val_viz_cnt)


def load_training_data(data_dir, data_set, caption_vector_length, n_classes) :
	if data_set == 'flowers' :
		flower_str_captions = pickle.load(
			open(join(data_dir, 'flowers', 'flowers_caps.pkl'), "rb"))

		img_classes = pickle.load(
			open(join(data_dir, 'flowers', 'flower_tc.pkl'), "rb"))

		flower_enc_captions = pickle.load(
			open(join(data_dir, 'flowers', 'flower_tv.pkl'), "rb"))
		tr_image_ids = pickle.load(
			open(join(data_dir, 'flowers', 'train_ids.pkl'), "rb"))
		val_image_ids = pickle.load(
			open(join(data_dir, 'flowers', 'val_ids.pkl'), "rb"))

		max_caps_len = caption_vector_length
		tr_n_imgs = len(tr_image_ids)
		val_n_imgs = len(val_image_ids)

		return {
			'image_list'    : tr_image_ids,
			'captions'      : flower_enc_captions,
			'data_length'   : tr_n_imgs,
			'classes'       : img_classes,
			'n_classes'     : n_classes,
			'max_caps_len'  : max_caps_len,
			'val_img_list'  : val_image_ids,
			'val_captions'  : flower_enc_captions,
			'val_data_len'  : val_n_imgs,
			'str_captions'  : flower_str_captions
		}

	else :
		raise Exception('No Dataset Found')


def initialize_directories(args):
	model_dir = join(args.data_dir, 'training', args.model_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	model_chkpnts_dir = join(model_dir, 'checkpoints')
	if not os.path.exists(model_chkpnts_dir):
		os.makedirs(model_chkpnts_dir)

	model_summaries_dir = join(model_dir, 'summaries')
	if not os.path.exists(model_summaries_dir):
		os.makedirs(model_summaries_dir)

	model_samples_dir = join(model_dir, 'samples')
	if not os.path.exists(model_samples_dir):
		os.makedirs(model_samples_dir)

	model_val_samples_dir = join(model_dir, 'val_samples')
	if not os.path.exists(model_val_samples_dir):
		os.makedirs(model_val_samples_dir)

	return model_dir, model_chkpnts_dir, model_samples_dir, \
		   model_val_samples_dir, model_summaries_dir


def save_for_viz_val(data_dir, generated_images, image_files, image_caps,
                     image_ids, image_size, id):

	generated_images = np.squeeze(np.array(generated_images))
	for i in range(0, generated_images.shape[0]) :
		image_dir = join(data_dir, str(image_ids[i]))
		if not os.path.exists(image_dir):
			os.makedirs(image_dir)

		real_image_path = join(image_dir,
							   '{}.jpg'.format(image_ids[i]))
		if os.path.exists(image_dir):
			real_images_255 = image_processing.load_image_array(image_files[i],
										image_size, image_ids[i], mode='val')
			scipy.misc.imsave(real_image_path, real_images_255)

		caps_dir = join(image_dir, "caps.txt")
		if not os.path.exists(caps_dir):
			with open(caps_dir, "w") as text_file:
				text_file.write(image_caps[i]+"\n")

		fake_images_255 = generated_images[i]
		scipy.misc.imsave(join(image_dir, 'fake_image_{}.jpg'.format(id)),
		                  fake_images_255)


def save_for_vis(data_dir, real_images, generated_images, image_files,
                 image_caps, image_ids) :

	shutil.rmtree(data_dir)
	os.makedirs(data_dir)

	for i in range(0, real_images.shape[0]) :
		real_images_255 = (real_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir,
			   '{}_{}.jpg'.format(i, image_files[i].split('/')[-1])),
		                  real_images_255)

		fake_images_255 = (generated_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir, 'fake_image_{}.jpg'.format(
			i)), fake_images_255)

	str_caps = '\n'.join(image_caps)
	str_image_ids = '\n'.join([str(image_id) for image_id in image_ids])
	with open(join(data_dir, "caps.txt"), "w") as text_file:
		text_file.write(str_caps)
	with open(join(data_dir, "ids.txt"), "w") as text_file:
		text_file.write(str_image_ids)


def get_val_caps_batch(batch_size, loaded_data, data_set, data_dir):

	if data_set == 'flowers':
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))

		batch_idx = np.random.randint(0, loaded_data['val_data_len'],
		                              size = batch_size)
		image_ids = np.take(loaded_data['val_img_list'], batch_idx)
		image_files = []
		image_caps = []
		for idx, image_id in enumerate(image_ids) :
			image_file = join(data_dir,
			                  'flowers/jpg/' + image_id)
			random_caption = random.randint(0, 4)
			captions[idx, :] = \
				loaded_data['val_captions'][image_id][random_caption][
				0 :loaded_data['max_caps_len']]

			image_caps.append(loaded_data['str_captions']
			                  [image_id][random_caption])
			image_files.append(image_file)

		return captions, image_files, image_caps, image_ids
	else:
		raise Exception('Dataset not found')


def get_training_batch(batch_no, batch_size, image_size, z_dim, split,
                       data_dir, data_set, loaded_data = None) :
	if data_set == 'flowers':
		real_images = np.zeros((batch_size, image_size, image_size, 3))
		wrong_images = np.zeros((batch_size, image_size, image_size, 3))
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		real_classes = np.zeros((batch_size, loaded_data['n_classes']))
		wrong_classes = np.zeros((batch_size, loaded_data['n_classes']))

		cnt = 0
		image_files = []
		image_caps = []
		image_ids = []
		for i in range(batch_no * batch_size,
		               batch_no * batch_size + batch_size) :
			idx = i % len(loaded_data['image_list'])
			image_file = join(data_dir,
			                  'flowers/jpg/' + loaded_data['image_list'][idx])

			image_ids.append(loaded_data['image_list'][idx])

			image_array = image_processing.load_image_array_flowers(image_file,
			                                                image_size)
			real_images[cnt, :, :, :] = image_array

			# Improve this selection of wrong image
			wrong_image_id = random.randint(0,
			                                len(loaded_data['image_list']) - 1)
			wrong_image_file = join(data_dir,
			                        'flowers/jpg/' + loaded_data['image_list'][
				                                            wrong_image_id])
			wrong_image_array = image_processing.load_image_array_flowers(wrong_image_file,
			                                                      image_size)
			wrong_images[cnt, :, :, :] = wrong_image_array
			
			wrong_classes[cnt, :] = loaded_data['classes'][loaded_data['image_list'][
									wrong_image_id]][0 :loaded_data['n_classes']]

			random_caption = random.randint(0, 4)
			captions[cnt, :] = \
			loaded_data['captions'][loaded_data['image_list'][idx]][
								random_caption][0 :loaded_data['max_caps_len']]

			real_classes[cnt, :] = \
				loaded_data['classes'][loaded_data['image_list'][idx]][
												0 :loaded_data['n_classes']]
			str_cap = loaded_data['str_captions'][loaded_data['image_list']
								[idx]][random_caption]

			image_files.append(image_file)
			image_caps.append(str_cap)
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files, \
		       real_classes, wrong_classes, image_caps, image_ids
	else:
		raise Exception('Dataset not found')


if __name__ == '__main__' :
	main()
