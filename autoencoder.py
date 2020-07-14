from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

def create_autoencoder() -> keras.Sequential:
    """
    :return: a convolutional autoencoder. Input: 256x256x1, Latent space: 16x16x1, Output:256x256x1
    """
    model = Sequential([
        Input((256, 256, 1)),
        Conv2D(16, (4, 4), padding='same', activation='relu'),
        MaxPool2D((4, 4)),
        Conv2D(8, (5, 5), padding='same', activation='relu'),
        MaxPool2D((4, 4)),
        Conv2D(1, (4, 4), padding='same', activation='relu'),

        Conv2D(1, (4, 4), padding='same', activation='relu'),
        UpSampling2D((4, 4)),
        Conv2D(8, (5, 5), padding='same', activation='relu'),
        UpSampling2D((4, 4)),
        Conv2D(16, (5, 5), padding='same', activation='relu'),
        Conv2D(1, (4, 4), padding='same', activation='sigmoid')
    ])
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


def loadDecoder(pathToModel: str = ".\\models\\35.h5") -> keras.Sequential:
    """
    :param pathToModel: path to the h5 autoencoder model.
    :return: Loads an autoencoder model, creates a generator model from the expanding layers, and returns it
    """
    fullAutoEncoder = load_model(pathToModel)
    decoder = Sequential()
    G = fullAutoEncoder.layers[5:]
    for layer in G:
        decoder.add(layer)
    decoder.build((None, 16, 16, 1))
    return decoder


def loadEncoder(pathToModel: str = ".\\models\\35.h5") -> keras.Sequential:
    """
    :param pathToModel: path to the h5 autoencoder model.
    :return: Loads an autoencoder model, creates an encoder model from the expanding layers, and returns it
    """
    fullAutoEncoder = load_model(pathToModel)
    encoder = Sequential()
    E = fullAutoEncoder.layers[:5]
    for layer in E:
        encoder.add(layer)
    encoder.build((None, 256, 256, 1))
    return encoder
