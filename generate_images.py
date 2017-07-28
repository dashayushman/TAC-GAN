import model
import argparse
import pickle
import scipy.misc
import random
import os

import tensorflow as tf
import numpy as np

from os.path import join


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

	parser.add_argument('--images_per_caption', type=int, default=30,
						help='The number of images that you want to generate '
	                         'per text description')

	parser.add_argument('--data_set', type=str, default="flowers",
						help='Dat set: MS-COCO, flowers')

	parser.add_argument('--checkpoints_dir', type=str, default="/tmp",
						help='Path to the checkpoints directory')


	args = parser.parse_args()

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

	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver(max_to_keep=10000)
	print('Trying to resume model from ' +
		  str(tf.train.latest_checkpoint(args.checkpoints_dir)))
	if tf.train.latest_checkpoint(args.checkpoints_dir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(args.checkpoints_dir))
		print('Successfully loaded model from ')
	else:
		print('Could not load checkpoints. Please provide a valid path to'
		      ' your checkpoints directory')
		exit()

	print('Starting to generate images from text descriptions.')
	for sel_i, text_cap in enumerate(loaded_data['text_caps']['features']):

		print('Text idx: {}\nRaw Text: {}\n'.format(sel_i, text_cap))
		captions_1, image_files_1, image_caps_1, image_ids_1,\
		image_caps_ids_1 = get_caption_batch(loaded_data, datasets_root_dir,
                         dataset=args.data_set, batch_size=args.batch_size)

		captions_1[args.batch_size-1, :] = text_cap

		for z_i in range(args.images_per_caption):
			z_noise = np.random.uniform(-1, 1, [args.batch_size, args.z_dim])
			val_feed = {
				input_tensors['t_real_caption'].name: captions_1,
				input_tensors['t_z'].name: z_noise,
				input_tensors['t_training'].name: True
			}

			val_gen = sess.run(
				[outputs['generator']],
				feed_dict=val_feed)
			dump_dir = os.path.join(args.data_dir,
			                        'images_generated_from_text')
			save_distributed_image_batch(dump_dir, val_gen, sel_i, z_i,
			                             args.batch_size)
	print('Finished generating images from text description')


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
		caps_new = pickle.load(
			open(join('Data', 'enc_text.pkl'), "rb"))

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
			'str_captions': flower_str_captions,
			'text_caps': caps_new
		}

	else:
		raise Exception('This dataset has not been handeled yet. '
		                 'Contributions are welcome.')


def save_distributed_image_batch(data_dir, generated_images, sel_i, z_i,
                                 batch_size=64):
	generated_images = np.squeeze(generated_images)
	folder_name = str(sel_i)
	image_dir = join(data_dir, folder_name)
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)
	fake_image_255 = generated_images[batch_size-1]
	scipy.misc.imsave(join(image_dir, '{}.jpg'.format(z_i)),
						  fake_image_255)


def get_caption_batch(loaded_data, data_dir, dataset='flowers', batch_size=64):

	captions = np.zeros((batch_size, loaded_data['max_caps_len']))
	batch_idx = np.random.randint(0, loaded_data['data_length'],
	                              size=batch_size)
	image_ids = np.take(loaded_data['image_list'], batch_idx)
	image_files = []
	image_caps = []
	image_caps_ids = []
	for idx, image_id in enumerate(image_ids):
		image_file = join(data_dir, dataset, 'jpg' + image_id)
		random_caption = random.randint(0, 4)
		image_caps_ids.append(random_caption)
		captions[idx, :] = \
			loaded_data['captions'][image_id][random_caption][
			0:loaded_data['max_caps_len']]

		image_caps.append(loaded_data['captions']
						  [image_id][random_caption])
		image_files.append(image_file)

	return captions, image_files, image_caps, image_ids, image_caps_ids

if __name__ == '__main__':
	main()