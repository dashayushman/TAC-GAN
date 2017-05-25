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

The following is the architecture of the TAC-GAN model

<img src="https://chalelele.files.wordpress.com/2017/05/tac-gan-1.png"
height="700" width="400" style="float:center">
