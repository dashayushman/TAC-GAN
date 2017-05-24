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

	parser.add_argument('--gfc_dim', type=int, default=1024,
						help='Dimension of gen untis for for fully connected '
							 'layer 1024')

	parser.add_argument('--caption_vector_length', type=int, default=256,
						help='Caption Vector Length')

	parser.add_argument('--data_dir', type=str, default="Data",
						help='Data Directory')


	parser.add_argument('--model_path', type=str, default=None,
						help='Pre-Trained Model Path, to resume from')

	parser.add_argument('--data_set', type=str, default="flowers",
						help='Dat set: MS-COCO, flowers')
	
	parser.add_argument('--caption_thought_vectors', type=str, default='Data/samples_features_dict.pkl',
                       help='Caption Thought Vector File')

	parser.add_argument('--n_images', type=int, default=5,
                       help='Number of Images per Caption')

	
	args = parser.parse_args()
	
	model_options = {
		'z_dim': args.z_dim,
		't_dim': args.t_dim,
		'batch_size': args.batch_size,
		'image_size': args.image_size,
		'gf_dim': args.gf_dim,
		'df_dim': args.df_dim,
		'gfc_dim': args.gfc_dim,
		'caption_vector_length': args.caption_vector_length,
		'n_classes': 102
	}

	gan = model.GAN(model_options)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess,  tf.train.latest_checkpoint(args.model_path))
	
	input_tensors, outputs = gan.build_generator()

	h = pickle.load( open(args.caption_thought_vectors , 'rb'))
	caption_vectors = h
	caption_image_dic = {}
	for cn, caption_vector in enumerate(caption_vectors):
		
		caption_images = []
		z_noise = np.random.uniform(-1, 1, [args.batch_size, args.z_dim])
		#caption = get_sampling_batch(args.batch_size, caption_vector)
		caption = [ caption_vector[0:args.caption_vector_length] ] * args.batch_size

		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )
		
		caption_images = [gen_image[i,:,:,:] for i in range(0, args.n_images)]
		caption_image_dic[ cn ] = caption_images
		print "Generated", cn

	for f in os.listdir( join(args.data_dir, 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join(args.data_dir, 'val_samples/' + f))

	for cn in range(0, len(caption_vectors)):
		caption_images = []
		for i, im in enumerate( caption_image_dic[ cn ] ):
			# im_name = "caption_{}_{}.jpg".format(cn, i)
			# scipy.misc.imsave( join(args.data_dir, 'val_samples/{}'.format(im_name)) , im)
			caption_images.append( im )
			caption_images.append( np.zeros((128, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		scipy.misc.imsave( join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)) , combined_image)


def get_sampling_batch(batch_size, caption):
	batch_encoder_inputs = []
	
	# Batch encoder inputs are just re-indexed encoder_inputs.
	for length_idx in xrange(max_len) :
		batch_encoder_inputs.append(
			np.array([[caption[length_idx]]
			          for batch_idx in xrange(batch_size)], dtype=np.float32))
	return batch_encoder_inputs

if __name__ == '__main__':
	main()
