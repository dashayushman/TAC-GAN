import numpy as np
from scipy import misc
import random
import skimage
import skimage.io
import skimage.transform
import os

def load_image_array_flowers(image_file, image_size):
	img = skimage.io.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'uint8')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = skimage.transform.resize(img, (image_size, image_size))

	# FLIP HORIZONTAL WIRH A PROBABILITY 0.5
	if random.random() > 0.5:
		img_resized = np.fliplr(img_resized)
	
	
	return img_resized.astype('float32')

def load_image_array(image_file, image_size,
					 image_id, data_dir='Data/datasets/mscoco/train2014',
					 mode='train'):
	img = None
	if os.path.exists(image_file):
		#print('found' + image_file)
		img = skimage.io.imread(image_file)
	else:
		print('notfound' + image_file)
		img = skimage.io.imread('http://mscoco.org/images/%d' % (image_id))
		img_path = os.path.join(data_dir, 'COCO_%s2014_%.12d.jpg' % ( mode,
																	  image_id))
		skimage.io.imsave(img_path, img)

	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'uint8')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = skimage.transform.resize(img, (image_size, image_size))

	# FLIP HORIZONTAL WIRH A PROBABILITY 0.5
	if random.random() > 0.5:
		img_resized = np.fliplr(img_resized)

	return img_resized.astype('float32')

def load_image_inception(image_file, image_size=128):
	img = skimage.io.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray((img.shape[0], img.shape[1], 3), dtype='uint8')
		img_new[:, :, 0] = img
		img_new[:, :, 1] = img
		img_new[:, :, 2] = img
		img = img_new

	if image_size != 0:
		img = skimage.transform.resize(img, (image_size, image_size), mode='reflect')

	return img.astype('int32')

if __name__ == '__main__':
	# TEST>>>
	arr = load_image_array('sample.jpg', 64)
	print(arr.mean())
	# rev = np.fliplr(arr)
	misc.imsave( 'rev.jpg', arr)
