# Littera
An implementation of a Deep Learning letter/character generator using Keras.




### Model
The Deep Learning model used is a simple Convolutional Autoencoder.
The Input and Output spaces are 256x256x1 images and the latent space is 16x16x1.
The model was trained on 28k+ Latin and Cyrillic characters.

### drawTool
I have implented a small pygame interface where you can draw a character by hand and see the reconstructed output of the model in real time. (pictures shown below)

Commands: LMouse=draw, RMouse = Erase, 'R' = reset/eraseAll

## Dataset
The dataset used to train the model is the crowdsourced [CoMNIST](https://github.com/GregVial/CoMNIST) dataset which contains the Latin and Cyrillic alphabets.
The dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

# Objectives
I am currently (Summer of 2020) learning Deep Learning using internet resources (outside my work in College) and I needed some practical project to demonstrate the knowledge gained so far.

I had the idea of generating characters for new languages/alphabets after remembering about the [Voynich Manuscript](https://en.wikipedia.org/wiki/Voynich_manuscript).

Ideally, I wanted a model that can create symbols that are character-like, and discernible from one another (the latter requiring some trial and error when generating them).

# Outcome and future plans

The model was able to recreate latin and cyrillic characters:

![Latin character T](/images/T.PNG)

![Cyrillic character J](/images/J.PNG)

Aswell as symbols outside the dataset, like numbers:

![number 3](/images/3.PNG)

The latent space (16x16x1) was apparently too large to only contain representation of character-like symbols, as I found that random scribbles are getting reconstructed adequately:

![random](/images/random.png)


### 20.7.13

My next plans are either to change the model into a Variational Autoencoder, or try using GANs.
Eitherways, this crude model brought a smile to my face when I first saw it reconstructing characters I had drawn.


