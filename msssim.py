#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of MS-SSIM.

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import os
import argparse
import sys

import tensorflow as tf
import numpy as np

from skimage.transform import resize
from scipy import signal
from scipy.ndimage.filters import convolve


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
  
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small 
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
  
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
  
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small 
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.
  
    Returns:
      MS-SSIM score between `img1` and `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
                im1, im2, max_val=max_val, filter_size=filter_size,
                filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [convolve(im, downsample_filter, mode='reflect')
                    for im in [im1, im2]]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) *
            (mssim[levels - 1] ** weights[levels - 1]))


def calculate_msssim(img_dir, gen_img_dir, caption_dir, output_dir):

    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    image_captions = {}
    image_classes = {}
    class_dirs = []
    class_names = []
    img_ids = []
    class_dict = {}
    gen_class_dict = {}

    print('Initializing objects for calculating MS-SSIM')
    for i in range(1, 103):
        class_dir_name = 'class_%.5d' % (i)
        class_dir = os.path.join(caption_dir, class_dir_name)
        class_names.append(class_dir_name)
        class_dirs.append(class_dir)
        onlyimgfiles = [f[0:11] + ".jpg" for f in os.listdir(class_dir)
                        if 'txt' in f]
        for img_file in onlyimgfiles:
            image_classes[img_file] = None

        for img_file in onlyimgfiles:
            image_captions[img_file] = []

    for class_dir, class_name in zip(class_dirs, class_names):
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        class_imgs = []
        gen_class_imgs = []
        for i, cap_file in enumerate(caption_files):
            if i % 50 == 0:
                print(str(i) + ' captions extracted from' + str(class_dir))

                class_imgs.append(cap_file[0:11] + ".jpg")
                image1_tr_path = os.path.join(gen_img_dir, 'train',
                                              cap_file[0:11] + ".jpg")
                if os.path.exists(image1_tr_path):
                    for root, subFolders, files in os.walk(image1_tr_path):
                        if files:
                            for f in files:
                                if 'jpg' in f:
                                    gen_class_imgs.append(os.path.join(root, f))

        class_dict[class_name] = class_imgs
        gen_class_dict[class_name] = gen_class_imgs
    with tf.Session() as sess:
        for class_name in class_dict.keys():
            img_list = class_dict[class_name]
            gen_img_list = gen_class_dict[class_name]
            real_msssim = []
            fake_msssim = []
            print('calculating MS-SSIM for real images of class : ' + str(
                class_name))
            for i in range(0, len(img_list)):
                for j in range(i, len(img_list)):
                    if (i == j):
                        continue
                    image1_path = os.path.join(img_dir, img_list[i])
                    image2_path = os.path.join(img_dir, img_list[j])
                    with open(image1_path, 'rb') as image_file:
                        img1_str = image_file.read()
                    with open(image2_path, 'rb') as image_file:
                        img2_str = image_file.read()
                    input_img = tf.placeholder(tf.string)
                    decoded_image = tf.expand_dims(
                            tf.image.decode_png(input_img, channels=3), 0)

                    img1 = np.squeeze(sess.run(decoded_image,
                                    feed_dict={input_img: img1_str}))
                    img2 = np.squeeze(sess.run(decoded_image,
                                    feed_dict={input_img: img2_str}))
                    img1 = resize(img1, (128, 128, 3), mode='reflect')
                    img2 = resize(img2, (128, 128, 3), mode='reflect')

                    img1 = np.expand_dims(img1, axis=0)
                    img2 = np.expand_dims(img2, axis=0)

                    real_msssim.append(MultiScaleSSIM(img1, img2, max_val=255))

            for i in range(0, len(gen_img_list)):
                for j in range(i, len(gen_img_list)):
                    if (i == j):
                        continue
                    image1_path = os.path.join('', gen_img_list[i])
                    image2_path = os.path.join('', gen_img_list[j])
                    with open(image1_path, 'rb') as image_file:
                        img1_str = image_file.read()
                    with open(image2_path, 'rb') as image_file:
                        img2_str = image_file.read()
                    input_img = tf.placeholder(tf.string)
                    decoded_image = tf.expand_dims(
                            tf.image.decode_png(input_img, channels=3), 0)
                    # with tf.Session() as sess:
                    img1 = sess.run(decoded_image,
                                    feed_dict={input_img: img1_str})
                    img2 = sess.run(decoded_image,
                                    feed_dict={input_img: img2_str})
                    fake_msssim.append(MultiScaleSSIM(img1, img2, max_val=255))

            mean_real_msssim = np.mean(real_msssim)
            mean_fake_msssim = np.mean(fake_msssim)

            tsv_dir = os.path.join(output_dir, 'msssim')
            tsv_path = os.path.join(tsv_dir, 'msssim.tsv')
            if not os.path.exists(tsv_dir):
                os.makedirs(tsv_dir)

            if os.path.exists(tsv_path):
                os.remove(tsv_path)

            with open(tsv_path, 'a') as f:
                str_real_mean = "%.9f" % mean_real_msssim
                str_fake_mean = "%.9f" % mean_fake_msssim
                f.write(
                    class_name + '\t' + str_real_mean + '\t' + str_fake_mean +
                    '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="Data/ms-ssim",
                        help='directory to dump all the images for '
                             'calculating inception score')

    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Root directory of the data')

    parser.add_argument('--dataset', type=str, default="flowers",
                        help='The root directory of the synthetic dataset')

    parser.add_argument('--syn_dataset_dir', type=str, default="flowers",
                        help='The root directory of the synthetic dataset')

    args = parser.parse_args()

    if args.dataset != 'flowers':
        print('Dataset Not Found')
        sys.exit()

    img_dir = os.path.join(args.data_dir, 'datasets', args.dataset, 'jpg')
    gen_img_dir = args.syn_dataset_dir
    caption_dir = os.path.join(args.data_dir, 'datasets', 'flowers',
                               'text_c10')

    calculate_msssim(img_dir, gen_img_dir, caption_dir, args.output_dir)
