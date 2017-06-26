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
Some important dependencies are the following and the rest can be installed 
using the ```requirements.txt```
1. Python 3.5
2. [Tensorflow 1.2.0](https://github.com/tensorflow/tensorflow)
4. [Theano 0.9.0](https://github.com/Theano/Theano) : for skip thought vectors
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

1. **Download the flower images** from
[here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz).
Extract the ```102flowers.tgz``` file and copy the extracted ```jpg``` folder
 to ```Data/datasets/flowers```

2. **Download the captions** from
[here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/).
Extract the downloaded file, copy the text_c10 folder and paste it in ```
Data/datasets/flowers``` directory

3. **Download the pretrained skip-thought vectors model** from
[here](https://github.com/ryankiros/skip-thoughts#getting-started) and copy
the downloaded files to ```Data/skipthoughts```

**NB:** *It is recommended to keep all the images in an SSD if available. This
 makes the batch loading and processing operation faster.*

### 3. Data Preprocessing
Extract the skip-thought features for the captions and prepare the dataset
for training by running the following script

```
python dataprep.py --data_dir=Data --dataset=flowers
```

This script will create a set of pickled files in the datet directory which
will be used during training.

### 4. Training


To train TAC-GAN with the default hyper parameters run the following script

```
python train.py --dataset="flowers"
```

the following flags can be set to change the hyperparameters of the network.

FLAG | VALUE TYPE | DEFAULT VALUE | DESCRIPTION
--- | --- | --- | ---
z-dim | int | 100 | Number of dimensions of the Noise vector |
t_dim | int | 512 | Number of dimensions for the latent representation of the text embedding.
batch_size | int | 64 | Mini-Batch Size.
image_size | int | 128 | Batch size to use during training.
gf_dim | int | 64 | Number of conv filters in the first layer of the generator.
df_dim | int | 64 | Number of conv filters in the first layer of the discriminator.
caption_vector_length | int | 4800 | Length of the caption vector embedding (vector generated using skip-thought vectors model).
n_classes | int | 102 | Number of classes
data_dir | String | Data | Data directory
learning_rate | float | 0.0002 | Learning rate
beta1 | float | 0.5 | Momentum for Adam Update
epochs | int | 200 | Maximum number of epochs to train.
save_every | int | 30 | Save model and samples after this many number.of iterations
resume_model | Boolean | False | Load pre-trained model
data_set | String | flowers | Which dataset to use: "flowers"
model_name | String | model_1 | Name of the model: Can be anything
train | bool | True | This is True while training and false otherwise. Used for batch normalization

We used the following script (hyper-parameters) in the paper

```
python train.py --t_dim=100 --image_size=128 --data_set=flowers --model_name=TAC_GAN --train=True --resume_model=True --z_dim=100 --n_classes=102 --epochs=400 --save_every=20 --caption_vector_length=4800 --batch_size=128
```



