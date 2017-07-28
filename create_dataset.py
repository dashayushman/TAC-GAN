import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import scipy.misc
import random
import os
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

	parser.add_argument('--n_classes', type=int, default=102,
	                    help='Number of classes/class labels')

	parser.add_argument('--data_dir', type=str, default="Data",
	                    help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
	                    help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
	                    help='Momentum for Adam Update')

	parser.add_argument('--epochs', type=int, default=200,
	                    help='Max number of epochs')

	parser.add_argument('--data_set', type=str, default="flowers",
	                    help='Dat set: flowers')

	parser.add_argument('--output_dir', type=str, default="Data/ds",
	                    help='The directory in which this dataset will be '
	                         'created')

	parser.add_argument('--checkpoints_dir', type=str, default="/tmp",
	                    help='Path to the checkpoints directory')

	args = parser.parse_args()

	model_stage_1_ds_tr, model_stage_1_ds_val, datasets_root_dir = \
														prepare_dirs(args)

	loaded_data = load_training_data(datasets_root_dir, args.data_set,
								 args.caption_vector_length, args.n_classes)

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

	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver(max_to_keep=10000)
	print('resuming model from checkpoint' +
	      str(tf.train.latest_checkpoint(args.checkpoints_dir)))
	if tf.train.latest_checkpoint(args.checkpoints_dir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(args.checkpoints_dir))
		print('Successfully loaded model from ')
	else:
		print('Could not load checkpoints')
		exit()

	print('Generating images for the captions in the training set at ' +
	      model_stage_1_ds_tr)
	for i in range(args.epochs):
		batch_no = 0
		while batch_no * args.batch_size + args.batch_size < \
				loaded_data['data_length']:

			real_images, wrong_images, caption_vectors, z_noise, image_files, \
			real_classes, wrong_classes, image_caps, image_ids, \
			image_caps_ids = get_training_batch(batch_no, args.batch_size,
					args.image_size, args.z_dim, datasets_root_dir,
					args.data_set, loaded_data)

			feed = {
				input_tensors['t_real_image'].name: real_images,
				input_tensors['t_wrong_image'].name: wrong_images,
				input_tensors['t_real_caption'].name: caption_vectors,
				input_tensors['t_z'].name: z_noise,
				input_tensors['t_real_classes'].name: real_classes,
				input_tensors['t_wrong_classes'].name: wrong_classes,
				input_tensors['t_training'].name: True
			}

			g_loss, gen = sess.run([loss['g_loss'], outputs['generator']],
			                       feed_dict=feed)

			print("LOSSES", g_loss, batch_no, i,
			      len(loaded_data['image_list']) / args.batch_size)
			batch_no += 1
			save_distributed_image_batch(model_stage_1_ds_tr, gen, image_caps,
							                        image_ids, image_caps_ids)

	print('Finished generating images for the training set captions.\n\n')
	print('Generating images for the captions in the validation set at ' +
	    model_stage_1_ds_val)
	for i in range(args.epochs):
		batch_no = 0
		while batch_no * args.batch_size + args.batch_size < \
				loaded_data['val_data_len']:

			val_captions, val_image_files, val_image_caps, val_image_ids, \
			val_image_caps_ids, val_z_noise = get_val_caps_batch(batch_no,
			        args.batch_size, args.z_dim, loaded_data, args.data_set,
                    datasets_root_dir)

			val_feed = {
				input_tensors['t_real_caption'].name: val_captions,
				input_tensors['t_z'].name: val_z_noise,
				input_tensors['t_training'].name: True
			}

			val_gen, val_attn_spn = sess.run(
				[outputs['generator'], checks['attn_span']],
				feed_dict=val_feed)

			print("LOSSES", batch_no, i, len(
				loaded_data['val_img_list']) / args.batch_size)
			batch_no += 1
			save_distributed_image_batch(model_stage_1_ds_val, val_gen,
										 val_image_caps,
										 val_image_ids,
										 val_image_caps_ids, val_attn_spn)
	print('Finished generating images for the validation set captions.\n\n')

def prepare_dirs(args):

	model_stage_1_ds_tr = join(args.output_dir, 'ds', 'train')
	if not os.path.exists(model_stage_1_ds_tr):
		os.makedirs(model_stage_1_ds_tr)

	model_stage_1_ds_val = join(args.output_dir, 'ds', 'val')
	if not os.path.exists(model_stage_1_ds_val):
		os.makedirs(model_stage_1_ds_val)

	datasets_root_dir = join(args.data_dir, 'datasets')

	return model_stage_1_ds_tr, model_stage_1_ds_val, datasets_root_dir


def load_training_data(data_dir, data_set, caption_vector_length, n_classes):
	if data_set == 'flowers':
		flower_str_captions = pickle.load(
			open(join(data_dir, 'flowers', 'flowers_caps.pkl'), "rb"))

		img_classes = pickle.load(
			open(join(data_dir, 'flowers', 'flower_tc.pkl'), "rb"))

		flower_enc_captions = pickle.load(
			open(join(data_dir, 'flowers', 'flower_tv.pkl'), "rb"))
		# h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
		tr_image_ids = pickle.load(
			open(join(data_dir, 'flowers', 'train_ids.pkl'), "rb"))
		val_image_ids = pickle.load(
			open(join(data_dir, 'flowers', 'val_ids.pkl'), "rb"))

		# n_classes = n_classes
		max_caps_len = caption_vector_length

		tr_n_imgs = len(tr_image_ids)
		val_n_imgs = len(val_image_ids)

		return {
			'image_list': tr_image_ids,
			'captions': flower_enc_captions,
			'data_length': tr_n_imgs,
			'classes': img_classes,
			'n_classes': n_classes,
			'max_caps_len': max_caps_len,
			'val_img_list': val_image_ids,
			'val_captions': flower_enc_captions,
			'val_data_len': val_n_imgs,
			'str_captions': flower_str_captions
		}

	else:
		raise Exception('Dataset not found')


def save_distributed_image_batch(data_dir, generated_images, image_caps,
					 image_ids, caps_ids):
	for i, (image_id, caps_id, image_cap) in enumerate(zip( image_ids, \
			caps_ids, image_caps)):
		image_dir = join(data_dir, str(image_id), str(caps_id))
		if not os.path.exists(image_dir):
			os.makedirs(image_dir)
		collection_dir = join(data_dir, 'collection')
		if not os.path.exists(collection_dir):
			os.makedirs(collection_dir)
		caps_dir = join(image_dir, "caps.txt")
		if not os.path.exists(caps_dir):
			with open(caps_dir, "w") as text_file:
				text_file.write(image_cap + "\n")

		fake_image_255 = (generated_images[i, :, :, :])
		if i == 0:
			scipy.misc.imsave(join(collection_dir, '{}.jpg'.format(image_id)),
							  fake_image_255)
		num_files = len(os.walk(image_dir).__next__()[2])
		scipy.misc.imsave(join(image_dir, '{}.jpg'.format(num_files + 1)),
						  fake_image_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim, data_dir,
                       data_set, loaded_data=None):

	if data_set == 'flowers':
		real_images = np.zeros((batch_size, image_size, image_size, 3))
		wrong_images = np.zeros((batch_size, image_size, image_size, 3))
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		real_classes = np.zeros((batch_size, loaded_data['n_classes']))
		wrong_classes = np.zeros((batch_size, loaded_data['n_classes']))

		cnt = 0
		image_files, image_caps, image_ids, image_caps_ids = [], [], [], []

		for i in range(batch_no * batch_size,
		               batch_no * batch_size + batch_size):

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
			wrong_image_array = image_processing.load_image_array_flowers(
				wrong_image_file,
				image_size)
			wrong_images[cnt, :, :, :] = wrong_image_array

			wrong_classes[cnt, :] = loaded_data['classes'][
										loaded_data['image_list'][
											wrong_image_id]][
									0:loaded_data['n_classes']]

			random_caption = random.randint(0, 4)
			image_caps_ids.append(random_caption)
			captions[cnt, :] = \
				loaded_data['captions'][loaded_data['image_list'][idx]][
					random_caption][0:loaded_data['max_caps_len']]

			real_classes[cnt, :] = \
				loaded_data['classes'][loaded_data['image_list'][idx]][
				0:loaded_data['n_classes']]
			str_cap = loaded_data['str_captions'][loaded_data['image_list']
			[idx]][random_caption]

			image_files.append(image_file)
			image_caps.append(str_cap)
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files, \
			   real_classes, wrong_classes, image_caps, image_ids, \
			   image_caps_ids
	else:
		raise Exception('Dataset not found')


def get_val_caps_batch(batch_no, batch_size, z_dim, loaded_data, data_set,
                       data_dir):

	if data_set == 'flowers':
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		batch_idx = range(batch_no * batch_size,
						  batch_no * batch_size + batch_size)
		image_ids = np.take(loaded_data['val_img_list'], batch_idx)

		image_files = []
		image_caps = []
		image_caps_ids = []

		for idx, image_id in enumerate(image_ids) :
			image_file = join(data_dir,
			                  'flowers/jpg/' + image_id)
			random_caption = random.randint(0, 4)
			image_caps_ids.append(random_caption)
			captions[idx, :] = \
				loaded_data['val_captions'][image_id][random_caption][
				0 :loaded_data['max_caps_len']]
			str_cap = loaded_data['str_captions'][image_id][random_caption]
			image_caps.append(loaded_data['str_captions']
			                  [image_id][random_caption])
			image_files.append(image_file)

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return captions, image_files, image_caps, image_ids, image_caps_ids, \
		       z_noise


if __name__ == '__main__':
	main()
