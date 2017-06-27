import os
import argparse
import progressbar

from shutil import copy
from Utils import inception_score as ins
from Utils import image_processing as ip



def prepare_inception_data(o_dir, i_dir):
	if not os.path.exists(o_dir):
		os.makedirs(o_dir)
		cnt = 0
		bar = progressbar.ProgressBar(redirect_stdout=True,
									  max_value=progressbar.UnknownLength)
		for root, subFolders, files in os.walk(i_dir):
			if files:
				for f in files:
					if 'jpg' in f:
						f_name = str(cnt) + '_ins.' + f.split('.')[-1]
						cnt += 1
						file_dir = os.path.join(root, f)
						dest_path = os.path.join(o_dir, f)
						dest_new_name = os.path.join(o_dir, f_name)
						copy(file_dir, o_dir)
						os.rename(dest_path, dest_new_name)
						bar.update(cnt)
		bar.finish()
		print('Total number of files: {}'.format(cnt))

def load_images(o_dir, i_dir, n_images=3000, size=128):
	prepare_inception_data(o_dir, i_dir)
	image_list = []
	done = False
	cnt = 0
	bar = progressbar.ProgressBar(redirect_stdout=True,
								  max_value=progressbar.UnknownLength)
	for root, dirs, files in os.walk(o_dir):
		if files:
			for f in files:
				cnt += 1
				file_dir = os.path.join(root, f)
				image_list.append(ip.load_image_inception(file_dir, 0))
				bar.update(cnt)
				if len(image_list) == n_images:
					done = True
					break
		if done:
			break
	bar.finish()
	print('Finished Loading Files')
	return image_list


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--output_dir', type=str, default="Data/ds_inception",
						help='directory to dump all the images for '
							 'calculating the inception score')

	parser.add_argument('--data_dir', type=str,
						default="Data/synthetic_dataset/ds",
						help='The root directory of the synthetic dataset')

	parser.add_argument('--n_images', type=int, default=30000,
						help='Number of images to consider for calculating '
							 'inception score')

	parser.add_argument('--image_size', type=int, default=128,
						help='Size of the image to consider for calculating '
							 'inception score')

	args = parser.parse_args()

	imgs_list = load_images(args.output_dir, args.data_dir,
							n_images=args.n_images, size=args.image_size)

	print('Extracting Inception Score')
	mean, std = ins.get_inception_score(imgs_list)
	print('Mean Inception Score: {}\nStandard Deviation: {}'.format(mean, std))