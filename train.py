import spacy
nlp = spacy.load('en')
import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
from pycocotools.coco import COCO



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
						help='Noise dimension')

	parser.add_argument('--t_dim', type=int, default=512,
						help='Text feature dimension')

	parser.add_argument('--batch_size', type=int, default=64,
						help='Batch Size')

	parser.add_argument('--image_size', type=int, default=64,
						help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64,
						help='Number of conv in the first layer gen.')

	parser.add_argument('--df_dim', type=int, default=64,
						help='Number of conv in the first layer discr.')

	parser.add_argument('--gfc_dim', type=int, default=1024,
						help='Dimension of gen untis for for fully connected '
							 'layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=4800,
						help='Caption Vector Length')

	parser.add_argument('--n_classes', type = int, default = 80,
	                    help = 'Number of classes/class labels')

	parser.add_argument('--attn_time_steps', type = int, default = 35,
	                    help = 'Number of time steps for attention model')

	parser.add_argument('--attn_word_feat_length', type = int, default = 300,
	                    help = 'Feature vector size for every word in the '
	                           'caption')

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

	parser.add_argument('--model_name', type=str, default="model_1",
						help='model_1 or model_2')

	parser.add_argument('--train', type = bool, default = True,
	                    help = 'True while training and False otherwise')

	args = parser.parse_args()

	model_dir = join(args.data_dir, 'training', args.model_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	model_chkpnts_dir = join(model_dir, 'checkpoints')
	if not os.path.exists(model_chkpnts_dir):
		os.makedirs(model_chkpnts_dir)

	model_samples_dir = join(model_dir, 'samples')
	if not os.path.exists(model_samples_dir):
		os.makedirs(model_samples_dir)

	model_val_samples_dir = join(model_dir, 'val_samples')
	if not os.path.exists(model_val_samples_dir):
		os.makedirs(model_val_samples_dir)

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
		'gfc_dim': args.gfc_dim,
		'caption_vector_length': args.caption_vector_length,
		'n_classes': loaded_data['n_classes'],
		'attn_time_steps' : args.attn_time_steps,
		'attn_word_feat_length' : args.attn_word_feat_length
	}

	gan = model.GAN(model_options)
	input_tensors, variables, loss, outputs, checks = gan.build_model()

	d_optim = tf.train.AdamOptimizer(args.learning_rate,
									 beta1=args.beta1).minimize(loss['d_loss'],
																var_list=
																variables[
																	'd_vars'])
	g_optim = tf.train.AdamOptimizer(args.learning_rate,
									 beta1=args.beta1).minimize(loss['g_loss'],
																var_list=
																variables[
																	'g_vars'])

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	saver = tf.train.Saver(max_to_keep=10000)
	if args.resume_model:
		print('resuming model from previous checkpoint' +
		      str(tf.train.latest_checkpoint(model_chkpnts_dir)))
		if tf.train.latest_checkpoint(model_chkpnts_dir) is not None:
			saver.restore(sess, tf.train.latest_checkpoint(model_chkpnts_dir))
			print('Successfully loaded model')
		else:
			print('Could not load checkpoints')

	for i in range(args.epochs):
		batch_no = 0
		while batch_no * args.batch_size + args.batch_size < \
				loaded_data['data_length']:

			real_images, wrong_images, caption_vectors, z_noise, image_files, \
			real_classes, wrong_classes, image_caps, image_ids, \
			captions_words_features  = get_training_batch(batch_no,
	                                              args.batch_size,
	                                              args.image_size,
	                                              args.z_dim,
	                                              'train',
	                                              datasets_root_dir,
	                                              args.data_set,
	                                              args.attn_time_steps,
	                                              args.attn_word_feat_length,
	                                              loaded_data)

			# DISCR UPDATE
			check_ts = [checks['d_loss1'], checks['d_loss2'],
							checks['d_loss3']]

			feed = {
				input_tensors['t_real_image'].name : real_images,
				input_tensors['t_wrong_image'].name : wrong_images,
				input_tensors['t_real_caption'].name : caption_vectors,
				input_tensors['t_z'].name : z_noise,
				input_tensors['t_real_classes'].name : real_classes,
				input_tensors['t_wrong_classes'].name : wrong_classes,
				input_tensors['t_training'].name : args.train
			}

			for c, d in zip(input_tensors['t_attn_input_seq'],
			                captions_words_features) :
				feed[c.name] = d

			_, d_loss, gen, d1, d2, d3 = sess.run(
				[d_optim, loss['d_loss'], outputs['generator']] + check_ts,
				feed_dict=feed)

			print "d1", d1
			print "d2", d2
			print "d3", d3
			print "D", d_loss

			# GEN UPDATE
			_, g_loss, gen, attn_spn = sess.run(
				[g_optim, loss['g_loss'], outputs['generator'],
				 checks['attn_span']],
				feed_dict=feed)

			# GEN UPDATE TWICE, to make sure d_loss does not go to 0
			_, g_loss, gen, attn_spn = sess.run(
				[g_optim, loss['g_loss'], outputs['generator'],
				 checks['attn_span']],
				feed_dict=feed)

			print "LOSSES", d_loss, g_loss, batch_no, i, len(
				loaded_data['image_list']) / args.batch_size
			#print attn_spn
			batch_no += 1
			if (batch_no % args.save_every) == 0:
				print "Saving Images, Model"

				save_for_vis(model_samples_dir, real_images, gen, image_files,
				             image_caps, image_ids, args.image_size)
				save_path = saver.save(sess,
				                       join(model_chkpnts_dir,
				                            "latest_model_{}_temp.ckpt".format(
										        args.data_set)))

				val_captions, val_image_files, val_image_caps, val_image_ids, \
				val_captions_words_features = get_val_caps_batch(args.batch_size,
		                                             loaded_data,
		                                             args.data_set,
		                                             'val',
		                                             datasets_root_dir,
		                                             args.attn_time_steps,
		                                             args.attn_word_feat_length)
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
					for c, d in zip(input_tensors['t_attn_input_seq'],
					                val_captions_words_features) :
						val_feed[c.name] = d
					val_gen, val_attn_spn = sess.run(
						[outputs['generator'], checks['attn_span']],
						feed_dict=val_feed)
					save_for_viz_val(model_val_samples_dir, val_gen,
					                 val_image_files, val_image_caps,
									 val_image_ids, args.image_size,
									 val_viz_cnt, val_attn_spn)

		if i % 1 == 0:
			epoch_dir = join(model_chkpnts_dir, str(i))
			if not os.path.exists(epoch_dir):
				os.makedirs(epoch_dir)
			save_path = saver.save(sess,
			                       join(epoch_dir,
			                            "model_after_{}_epoch_{}.ckpt".
			                                format(args.data_set, i)))
			val_captions, val_image_files, val_image_caps, val_image_ids, \
			val_captions_words_features = get_val_caps_batch(args.batch_size,
		                                             loaded_data,
		                                             args.data_set,
		                                             'val',
		                                             datasets_root_dir,
		                                             args.attn_time_steps,
		                                             args.attn_word_feat_length)
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
				for c, d in zip(input_tensors['t_attn_input_seq'],
				                val_captions_words_features) :
					val_feed[c.name] = d
				val_gen, val_attn_spn = sess.run(
					[outputs['generator'], checks['attn_span']],
					feed_dict=val_feed)
				save_for_viz_val(model_val_samples_dir, val_gen,
				                 val_image_files, val_image_caps,
								 val_image_ids, args.image_size,
								 val_viz_cnt, val_attn_spn)

def load_training_data(data_dir, data_set, caption_vector_length, n_classes) :
	if data_set == 'flowers' :
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
		tr_caps_features = pickle.load(
			open(os.path.join(data_dir, 'mscoco/train', 'coco_tr_tv.pkl'),
				 	"rb"))

		tr_img_classes = pickle.load(
			open(os.path.join(data_dir, 'mscoco/train', 'coco_tr_tc.pkl'),
				 "rb"))

		val_caps_features = pickle.load(
			open(os.path.join(data_dir, 'mscoco/val', 'coco_tr_tv.pkl'),
				 "rb"))

		# n_classes = 80
		max_caps_len = caption_vector_length
		tr_annFile = '%s/annotations_inst/instances_%s.json' % (
								join(data_dir, 'mscoco'), 'train2014')
		tr_annFile_caps = '%s/annotations_caps/captions_%s.json' % \
		                  (join(data_dir, 'mscoco'), 'train2014')

		val_annFile = '%s/annotations_inst/instances_%s.json' % (
			join(data_dir, 'mscoco'), 'val2014')
		val_annFile_caps = '%s/annotations_caps/captions_%s.json' % \
		                   (join(data_dir, 'mscoco'), 'val2014')

		val_caps_coco = COCO(val_annFile_caps)
		val_coco = COCO(val_annFile)

		val_img_list = val_coco.getImgIds()
		val_n_imgs = len(val_img_list)

		tr_caps_coco = COCO(tr_annFile_caps)
		tr_coco = COCO(tr_annFile)

		tr_image_list = tr_coco.getImgIds()
		tr_n_imgs = len(tr_image_list)
		return {
			'image_list': tr_image_list,
			'captions': tr_caps_features,
			'data_length': tr_n_imgs,
			'classes': tr_img_classes,
			'n_classes': n_classes,
			'max_caps_len': max_caps_len,
			'tr_coco_obj' : tr_coco,
			'tr_coco_caps_obj' : tr_caps_coco,
			'val_coco_obj': val_coco,
			'val_coco_caps_obj': val_caps_coco,
			'val_img_list': val_img_list,
			'val_captions' : val_caps_features,
			'val_data_len' : val_n_imgs
		}

def save_for_viz_val(data_dir, generated_images, image_files, image_caps,
                     image_ids, image_size, id, val_attn_spn):

	for i in range(0, generated_images.shape[0]) :
		image_dir = join(data_dir, str(image_ids[i]))
		if not os.path.exists(image_dir):
			os.makedirs(image_dir)

		real_image_path = join(image_dir,
							   '{}.jpg'.format(image_ids[i]))
		if os.path.exists(image_dir):
			real_images_255 = image_processing.load_image_array(image_files[i],
														image_size,
														image_ids[i],
														data_dir='Data/datasets'
																 '/mscoco'
																 '/val2014',
														mode='val')
			scipy.misc.imsave(real_image_path, real_images_255)

		caps_dir = join(image_dir, "caps.txt")
		if not os.path.exists(caps_dir):
			with open(caps_dir, "w") as text_file:
				text_file.write(image_caps[i]+"\n")
				text_file.write("\t".join(["{}".format(val_attn_) for
										   val_attn_ in val_attn_spn[i]]))

		fake_images_255 = (generated_images[i, :, :, :])
		scipy.misc.imsave(join(image_dir, 'fake_image_{}.jpg'.format(id)),
		                  fake_images_255)

def save_for_vis(data_dir, real_images, generated_images, image_files,
                 image_caps, image_ids, image_size) :
	shutil.rmtree(data_dir)
	os.makedirs(data_dir)

	for i in range(0, real_images.shape[0]) :
		real_image_255 = np.zeros((image_size, image_size, 3), dtype = np.uint8)
		real_images_255 = (real_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir,
			   '{}_{}.jpg'.format(i, image_files[i].split('/')[-1])),
		                  real_images_255)

		fake_image_255 = np.zeros((image_size, image_size, 3), dtype = np.uint8)
		fake_images_255 = (generated_images[i, :, :, :])
		scipy.misc.imsave(join(data_dir, 'fake_image_{}.jpg'.format(
			i)), fake_images_255)
	str_caps = '\n'.join(image_caps)
	str_image_ids = '\n'.join([str(image_id) for image_id in image_ids])
	with open(join(data_dir, "caps.txt"), "w") as text_file:
		text_file.write(str_caps)
	with open(join(data_dir, "ids.txt"), "w") as text_file:
		text_file.write(str_image_ids)

def get_val_caps_batch(batch_size, loaded_data, data_set, split, data_dir,
                       attn_time_steps, attn_word_feat_length):
	if data_set == 'mscoco':
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		captions_words_features = np.zeros((batch_size, attn_time_steps,
		                                    attn_word_feat_length))
		batch_idx = np.random.randint(0, loaded_data['val_data_len'],
									  size=batch_size)
		image_ids = np.take(loaded_data['val_img_list'], batch_idx)
		image_files = []
		image_caps = []
		for idx, image_id in enumerate(image_ids):
			image_file = join(data_dir, 'mscoco/%s2014/COCO_%s2014_%.12d.jpg' % (
				split, split, image_id))
			random_caption = random.randint(0, 4)
			captions[idx, :] = loaded_data['val_captions'][image_id][
			                   random_caption][0:loaded_data['max_caps_len']]
			annIds_ = loaded_data['val_coco_caps_obj'].getAnnIds(imgIds=image_id)
			anns = loaded_data['val_coco_caps_obj'].loadAnns(annIds_)
			img_caps = [ann['caption'] for ann in anns]
			str_cap = img_caps[random_caption]

			unicode_cap_str = str_cap.decode('utf-8')
			spacy_cap_obj = nlp(unicode_cap_str)
			word_feats = None
			for i, tok in enumerate(spacy_cap_obj) :
				if i >= attn_time_steps :
					break
				if word_feats is None :
					word_feats = [tok.vector]
				else:
					word_feats = np.concatenate((word_feats, [tok.vector]),
				                            axis = 0)
			pad_len = attn_time_steps - len(spacy_cap_obj)
			if pad_len > 0 :
				pad_vecs = np.zeros((pad_len, attn_word_feat_length))
				word_feats = np.concatenate((word_feats, pad_vecs), axis = 0)

			captions_words_features[idx, :, :] = word_feats
			image_caps.append(img_caps[random_caption])
			image_files.append(image_file)

		captions_words_features = np.transpose(captions_words_features,
		                                       (1, 0, 2))
		captions_words_features = np.split(captions_words_features,
		                                   attn_time_steps)

		for i in range(0, len(captions_words_features)) :
			captions_words_features[i] = np.squeeze(captions_words_features[i])

		#z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return captions, image_files, image_caps, image_ids,\
		       captions_words_features
	elif data_set == 'flowers':
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		captions_words_features = np.zeros((batch_size, attn_time_steps,
		                                    attn_word_feat_length))
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
			str_cap = loaded_data['str_captions'][image_id][random_caption]
			unicode_cap_str = str_cap.decode('utf-8')
			spacy_cap_obj = nlp(unicode_cap_str)
			word_feats = None
			for i, tok in enumerate(spacy_cap_obj) :
				if i >= attn_time_steps :
					break
				if word_feats is None :
					word_feats = [tok.vector]
				else:
					word_feats = np.concatenate((word_feats, [tok.vector]),
				                            axis = 0)
			pad_len = attn_time_steps - len(spacy_cap_obj)
			if pad_len > 0 :
				pad_vecs = np.zeros((pad_len, attn_word_feat_length))
				word_feats = np.concatenate((word_feats, pad_vecs), axis = 0)

			captions_words_features[idx, :, :] = word_feats
			image_caps.append(loaded_data['str_captions']
			                  [image_id][random_caption])
			image_files.append(image_file)

		captions_words_features = np.transpose(captions_words_features,
		                                       (1, 0, 2))
		captions_words_features = np.split(captions_words_features,
		                                   attn_time_steps)
		for i in range(0, len(captions_words_features)) :
			captions_words_features[i] = np.squeeze(captions_words_features[i])

		#z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return captions, image_files, image_caps, image_ids,\
		       captions_words_features

def get_training_batch(batch_no, batch_size, image_size, z_dim, split,
                       data_dir, data_set, attn_time_steps,
                       attn_word_feat_length, loaded_data = None) :
	if data_set == 'mscoco' :

		real_images = np.zeros((batch_size, image_size , image_size, 3))
		wrong_images = np.zeros((batch_size, image_size, image_size, 3))
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		real_classes = np.zeros((batch_size, loaded_data['n_classes']))
		wrong_classes = np.zeros((batch_size, loaded_data['n_classes']))
		captions_words_features = np.zeros((batch_size, attn_time_steps,
		                                    attn_word_feat_length))
		img_range = range(batch_no * batch_size,
						  batch_no * batch_size + batch_size)
		#batch_idx = np.random.randint(0, loaded_data['data_length'],
		#							  size=batch_size)
		image_ids = np.take(loaded_data['image_list'], img_range)
		image_files = []
		image_caps = []
		'''
		for i in range(batch_no * batch_size,
					   batch_no * batch_size + batch_size):
			idx = i % len(loaded_data['image_list'])
		'''
		for idx, image_id in enumerate(image_ids) :
			image_file = join(data_dir, 'mscoco/%s2014/COCO_%s2014_%.12d.jpg' % (
				split, split, image_id))
			image_array = image_processing.load_image_array(image_file,
			                                                	image_size,
															image_id)
			real_images[idx, :, :, :] = image_array

			random_caption = random.randint(0, 4)


			captions[idx, :] = \
				loaded_data['captions'][image_id][random_caption][0:loaded_data['max_caps_len']]
			#print('i: ' + str(idx) + '\tcaps vec: ' + str(loaded_data['classes'][image_id]))
			if type(loaded_data['classes'][image_id]) == np.ndarray:
				real_classes[idx, :] = \
					loaded_data['classes'][image_id][0:loaded_data['n_classes']]
			else:
				real_classes[idx, :] = np.zeros(loaded_data['n_classes'])
				#print('case')

			annIds_ = loaded_data['tr_coco_caps_obj'].getAnnIds(imgIds=image_id)
			anns = loaded_data['tr_coco_caps_obj'].loadAnns(annIds_)
			img_caps = [ann['caption'] for ann in anns]
			str_cap = img_caps[random_caption]

			unicode_cap_str = str_cap.decode('utf-8')
			spacy_cap_obj = nlp(unicode_cap_str)
			word_feats = None

			for i, tok in enumerate(spacy_cap_obj):
				if i >= attn_time_steps:
					break
				if word_feats is None:
					word_feats = [tok.vector]
				else:
					word_feats = np.concatenate((word_feats, [tok.vector]),
											  axis=0)
			pad_len = attn_time_steps - len(spacy_cap_obj)
			if pad_len > 0:
				pad_vecs = np.zeros((pad_len, attn_word_feat_length))
				word_feats = np.concatenate((word_feats, pad_vecs), axis = 0)
			#print(idx,word_feats.shape, len(spacy_cap_obj))
			captions_words_features[idx, :, :] = word_feats

			image_caps.append(str_cap)
			image_files.append(image_file)

		captions_words_features = np.transpose(captions_words_features,
			                                       (1, 0, 2))
		captions_words_features = np.split(captions_words_features,
		                                   attn_time_steps)
		for i in range(0, len(captions_words_features)) :
			captions_words_features[i] = np.squeeze(captions_words_features[i])

		# TODO>> As of Now, wrong images are just shuffled real images.
		first_image = real_images[0, :, :, :]
		first_class = real_classes[0, : ]
		for i in range(0, batch_size) :
			if i < batch_size - 1 :
				wrong_images[i, :, :, :] = real_images[i + 1, :, :, :]
				wrong_classes[i, :] = real_classes[i + 1, :]
			else :
				wrong_images[i, :, :, :] = first_image
				wrong_classes[i, :] = first_class

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

		return real_images, wrong_images, captions, z_noise, image_files, \
			   real_classes, wrong_classes, image_caps, image_ids,\
			   captions_words_features

	if data_set == 'flowers':
		real_images = np.zeros((batch_size, image_size, image_size, 3))
		wrong_images = np.zeros((batch_size, image_size, image_size, 3))
		captions = np.zeros((batch_size, loaded_data['max_caps_len']))
		real_classes = np.zeros((batch_size, loaded_data['n_classes']))
		wrong_classes = np.zeros((batch_size, loaded_data['n_classes']))
		captions_words_features = np.zeros((batch_size, attn_time_steps,
		                                    attn_word_feat_length))

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
			unicode_cap_str = str_cap.decode('utf-8')
			spacy_cap_obj = nlp(unicode_cap_str)
			word_feats = None
			for i, tok in enumerate(spacy_cap_obj) :
				if i >= attn_time_steps :
					break
				if word_feats is None :
					word_feats = [tok.vector]
				else:
					word_feats = np.concatenate((word_feats, [tok.vector]),
				                            axis = 0)
			pad_len = attn_time_steps - len(spacy_cap_obj)
			if pad_len > 0 :
				pad_vecs = np.zeros((pad_len, attn_word_feat_length))
				word_feats = np.concatenate((word_feats, pad_vecs), axis = 0)

			captions_words_features[cnt, :, :] = word_feats

			image_files.append(image_file)
			image_caps.append(str_cap)
			cnt += 1

		captions_words_features = np.transpose(captions_words_features,
		                                       (1, 0, 2))
		captions_words_features = np.split(captions_words_features,
		                                   attn_time_steps)
		for i in range(0, len(captions_words_features)) :
			captions_words_features[i] = np.squeeze(captions_words_features[i])

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files, \
		       real_classes, wrong_classes, image_caps, image_ids,\
		       captions_words_features

def tf_seq_reshape(batch_size, captions, caps_max_len):
	# Now we create batch-major vectors from the data selected above.
	batch_encoder_inputs = []

	# Batch encoder inputs are just re-indexed encoder_inputs.
	for length_idx in xrange(caps_max_len) :
		batch_encoder_inputs.append(
			np.array([[captions[batch_idx][length_idx]]
			          for batch_idx in xrange(batch_size)], dtype = np.float32))
	return batch_encoder_inputs

if __name__ == '__main__' :
	main()
