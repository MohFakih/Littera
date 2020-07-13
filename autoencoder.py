from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D


def create_autoencoder() -> keras.Sequential:
    """
    :return: a convolutional autoencoder. Input: 256x256x1, Latent space: 16x16x1, Output:256x256x1
    """
    model = keras.Sequential([
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

A = create_autoencoder()
A.summary()