![TAC-GAN](https://chalelele.files.wordpress.com/2017/05/logo.png)

This is the official implementation of the TAC-GAN model presented in
[https://arxiv.org/abs/1703.06412](https://arxiv.org/abs/1703.06412).

Text Conditioned Auxiliary Classifier Generative Adversarial Network,
(TAC-GAN) is a text to image Generative Adversarial Network (GAN) for
synthesizing images from their text descriptions. TAC-GAN builds upon the
[AC-GAN](https://arxiv.org/abs/1610.09585) by conditioning the generated images
on a text description instead of on a class label. In the presented TAC-GAN
model, the input vector of the Generative network is built based on a noise
vector and another vector containing an embedded representation of the
textual description. While the Discriminator is similar to that of
the AC-GAN, it is also augmented to receive the text information as
input before performing its classification.

For embedding the textual descriptions of the images into vectors we used
[skip-thought vectors](https://arxiv.org/abs/1506.06726)

The following is the architecture of the TAC-GAN model

<img src="https://chalelele.files.wordpress.com/2017/05/tac-gan-1.png"
height="700" width="400" style="float:center">

# Prerequisites

1. Python 2.7.6
2. [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow)
3. [h5py 2.6.0](http://www.h5py.org/)
4. [Theano 0.8.2](https://github.com/Theano/Theano) : for skip thought vectors
5. [scikit-learn](http://scikit-learn.org/stable/index.html) : for skip thought vectors
6. [NLTK 3.2.1](http://www.nltk.org/) : for skip thought vectors

It is recommended to use a virtual environment for running this project and
installing the required dependencies in it by using the
[***requirements.txt***](https://github.com/dashayushman/TAC-GAN/blob/master/requirements.txt) file.

The project has been tested on a Ubuntu 14.04 machine with an 12 GB NVIDIA
Titen X GPU

# Setup and Run

### 1. Clone the Repository

```
git clone https://github.com/dashayushman/TAC-GAN.git
cd TAC-GAN
```

### 2. Download the Dataset

The model presented in the paper was trained on the
[flowers dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ ). This
To train the TAC-GAN on the flowers dataset, first, download the dataset by
doing the following,

1. Download the flower images from
[here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz).
Extract the ```102flowers.tgz``` file and copy the extracted ```jpg``` folder
 to the following directory.

    ```
    Data
      |__datasets
             |___flowers
    ```

2. Download the captions from
[here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/).
Extract the downloaded file, copy the text_c10 folder and paste it in ```
Data/datasets/flowers``` directory




