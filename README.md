# Littera
Different implementation(s) of a Deep Learning letter/character generator using TF/Keras.




### Autoencoder
The first model used is a simple Convolutional Autoencoder.

The Input and Output spaces are (256, 256, 1) images and the latent space is a (16, 16, 1) Tensor.

The model was trained on 28k+ Latin and Cyrillic characters.

###GAN

The second model implemented is a GAN.

The Output space is a (64, 64, 1) image.

The Latent/Noise space is a (16, 1) tensor.


### drawTool showcase
I have implented a small pygame interface where you can draw a character by hand and see the reconstructed output of the autoencoder model in real time. (pictures shown below)

Commands: LMouse=draw, RMouse = Erase, 'R' = reset/eraseAll

### animation showcase
3D Simplex noise is used to smoothly move through the latent space in order to create an "animation". The 16x16 latent space and the 256x256 output are visualized next to each other.

##### Both showcases mentioned are in /showcases
##### The notebooks are essentially only used to train and manually verify the model

## Dataset
The dataset used to train the model is the crowdsourced [CoMNIST](https://github.com/GregVial/CoMNIST) dataset which contains the Latin and Cyrillic alphabets.
The dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

# Objectives
I am currently (Summer of 2020) learning Deep Learning using internet resources (outside my work in College) and I needed some practical project to demonstrate the knowledge gained so far.

I had the idea of generating characters for new languages/alphabets after remembering about the [Voynich Manuscript](https://en.wikipedia.org/wiki/Voynich_manuscript).

Ideally, I wanted a model that can create symbols that are character-like, and discernible from one another (the latter requiring some trial and error when generating them).

# Outcome and future plans

The autoencoder model was able to recreate latin and cyrillic characters:

![Latin character T](/images/T.PNG)

![Cyrillic character J](/images/J.PNG)

Aswell as symbols outside the dataset, like numbers:

![number 3](/images/3.PNG)

The latent space (16x16x1) was apparently too large to only contain representation of character-like symbols, as I found that random scribbles are getting reconstructed adequately:

![random](/images/random.PNG)

### 13.7.20

My next plans are either to change the model into a Variational Autoencoder, or try using GANs. Eitherways, this crude model brought a smile to my face when I first saw it reconstructing characters I had drawn.

### 14.7.20

My initial next move was to implement some form of Variational Auto-Encoder, but I had doubts that the Auto-Encoder model was just downscaling and upscaling the characters.
And after building the "animation showcase", my fears were validated. My next step will probably be GANs.

### 19.7.20

It took me a couple of days of research to be comfortable enough to hand-code GANs from scratch.
Training is way slower compared to the auto-encoder, even when resorting to public accelerators (Google/Kaggle GPUs).
My puny 1060-3Gb could not handle the size of the tensors required to train on 256x256 images, so I had to downscale the images to 64x64.

I also used Conv2DTranspose layers instead of normal upsampling in order to allow more flexibility in generation.

I resorted to many methods to speed up training (BatchNormalization, LeakyReLU instead of ReLU for the discriminator, Different learning rates for the generator and the discriminator,
switching between training the discriminator and the generator until each achieves good validation accuracy, using optimizers with variable learning rates, changing batch sizes), but I could not achieve
a decent model even after substantial amounts of training (3k+ epochs).

### I will only keep the recent plans here and I will move them to a different text file when they get too numerous.
