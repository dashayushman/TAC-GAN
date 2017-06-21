import os
from shutil import copyfile, copy
from Utils import inception_score as ins
from Utils import image_processing as ip



def prepare_inception_data(root_dir):


	if not os.path.exists(root_dir):
		os.makedirs(root_dir)
		cnt = 0
		for root, subFolders, files in os.walk(
				'Data/training/model_15_ds_flowers_stage_2_256/stage_2_ds'):
			if files:
				for f in files:
					if 'jpg' in f:
						f_name = str(cnt) + '_ins.' + f.split('.')[-1]
						cnt += 1
						print(cnt)
						file_dir = os.path.join(root, f)
						dest_path = os.path.join(root_dir, f)
						dest_new_name = os.path.join(root_dir, f_name)
						copy(file_dir, root_dir)
						os.rename(dest_path, dest_new_name)
		print cnt

def load_images(root_dir, n_images=80000, size=128):
	image_list = []
	done = False
	for root, dirs, files in os.walk(root_dir):
		if files:
			for f in files:
				file_dir = os.path.join(root, f)
				image_list.append(ip.load_image_inception(file_dir, 0))
				if len(image_list) == n_images:
					done = True
					break
		if done:
			break
	return image_list


if __name__ == '__main__':
	dump_dir = 'Data/training/model_15_ds_flowers_stage_2_256' \
			   '/stage_1_ds_inception'
	prepare_inception_data(dump_dir)
	imgs_list = load_images(dump_dir, n_images=30000, size=256)
	mean, std = ins.get_inception_score(imgs_list)
	print(mean, std)