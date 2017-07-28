import model
import argparse
import pickle
import scipy.misc
import random
import os
import progressbar

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

    parser.add_argument('--data_set', type=str, default="flowers",
                        help='Dat set: flowers')

    parser.add_argument('--output_dir', type=str,
                        default="Data/synthetic_dataset",
                        help='The directory in which this dataset will be '
                             'created')

    parser.add_argument('--checkpoints_dir', type=str, default="/tmp",
                        help='Path to the checkpoints directory')

    parser.add_argument('--n_interp', type=int, default=100,
                        help='The difference between each interpolation. '
                             'Should ideally be a multiple of 10')

    parser.add_argument('--n_images', type=int, default=500,
                        help='Number of images to randomply sample for '
                             'generating interpolation results')

    args = parser.parse_args()
    datasets_root_dir = join(args.data_dir, 'datasets')

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
        print('Successfully loaded model')
    else:
        print('Could not load checkpoints')
        exit()

    random.shuffle(loaded_data['image_list'])
    selected_images = loaded_data['image_list'][:args.n_images]
    cap_id = [np.random.randint(0, 4) for cap_i in range(len(selected_images))]

    print('Generating Images by interpolating z')
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                  max_value=args.n_images)
    for sel_i, (sel_img, sel_cap) in enumerate(zip(selected_images, cap_id)):
        captions, image_files, image_caps, image_ids, image_caps_ids = \
            get_images_z_intr(sel_img, sel_cap, loaded_data,
                              datasets_root_dir, args.batch_size)

        z_noise_1 = np.full((args.batch_size, args.z_dim), -1.0)
        z_noise_2 = np.full((args.batch_size, args.z_dim), 1.0)
        intr_z_list = get_interp_vec(z_noise_1, z_noise_2, args.z_dim,
                                     args.n_interp, args.batch_size)

        for z_i, z_noise in enumerate(intr_z_list):
            val_feed = {
                input_tensors['t_real_caption'].name: captions,
                input_tensors['t_z'].name: z_noise,
                input_tensors['t_training'].name: True
            }

            val_gen = sess.run([outputs['generator']], feed_dict=val_feed)

            save_distributed_image_batch(args.output_dir, val_gen, sel_i, z_i,
                                         sel_img, sel_cap, args.batch_size)
        bar.update(sel_i)
    bar.finish()
    print('Finished generating interpolated images')


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
        raise Exception('Dataset Not Found!!')

def save_distributed_image_batch(data_dir, generated_images, sel_i, z_i,
                                 sel_img, sel_cap, batch_size):

    generated_images = np.squeeze(generated_images)
    image_dir = join(data_dir, 'z_interpolation', str(sel_i))
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    meta_path = os.path.join(image_dir, "meta.txt")
    with open(meta_path, "w") as text_file:
        text_file.write(str(sel_img) + "\t" + str(sel_cap))
    fake_image_255 = generated_images[batch_size - 1]
    scipy.misc.imsave(join(image_dir, '{}.jpg'.format(z_i)),
                      fake_image_255)


def get_images_z_intr(sel_img, sel_cap, loaded_data, data_dir, batch_size=64):

    captions = np.zeros((batch_size, loaded_data['max_caps_len']))
    batch_idx = np.random.randint(0, loaded_data['data_length'],
                                 size = batch_size-1)

    image_ids = np.take(loaded_data['image_list'], batch_idx)
    image_files = []
    image_caps = []
    image_caps_ids = []

    for idx, image_id in enumerate(image_ids):
        image_file = join(data_dir,
                          'flowers/jpg/' + image_id)
        random_caption = random.randint(0, 4)
        image_caps_ids.append(random_caption)
        captions[idx, :] = \
            loaded_data['captions'][image_id][random_caption][
            0:loaded_data['max_caps_len']]
        str_cap = loaded_data['str_captions'][image_id][random_caption]

        image_caps.append(loaded_data['captions']
                          [image_id][random_caption])
        image_files.append(image_file)
        if idx == batch_size-2:
            idx = idx+1
            image_id = sel_img
            image_file = join(data_dir,
                              'flowers/jpg/' + sel_img)
            random_caption = sel_cap
            image_caps_ids.append(random_caption)
            captions[idx, :] = \
                loaded_data['captions'][image_id][random_caption][
                0:loaded_data['max_caps_len']]
            str_cap = loaded_data['str_captions'][image_id][random_caption]
            image_caps.append(loaded_data['str_captions']
                              [image_id][random_caption])
            image_files.append(image_file)
            break

    return captions, image_files, image_caps, image_ids, image_caps_ids


def get_interp_vec(vec_1, vec_2, dim, n_interp, batch_size):

    intrip_list = []
    bals = np.arange(0, 1, 1/n_interp)
    for bal in bals:
        left = np.full((batch_size, dim), bal)
        right = np.full((batch_size, dim), 1.0 - bal)
        intrip_vec = np.multiply(vec_1, left) + np.multiply(vec_2, right)
        intrip_list.append(intrip_vec)
    return intrip_list


if __name__ == '__main__':
    main()
