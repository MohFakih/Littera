import pygame as pg
import numpy as np
from opensimplex import OpenSimplex
from pgUtilities.PGWindow import PGWindow
from autoencoder import loadDecoder
from math import cos, pi

class animationDisplay(PGWindow):
    def __init__(self):
        super().__init__()

        # Load the tensorflow model
        self.model = loadDecoder()

        # All the necessary initializations
        self.reset()

    def handleEvent(self, event: pg.event) -> None:
        pass

    def reset(self):
        # Pixel matrices
        self.latentSpaceArray = np.zeros((16, 16))
        self.pixelArray = np.zeros((512, 512 * 2, 3))

        # simplex noise generator
        self.seed = 600613  # Google <3
        self.noiseGenerator = OpenSimplex(self.seed)
        self.NoiseX = self.seed/7919  # Noise-Space coordinates, need any non-integer value
        self.NoiseY = self.seed/7919  # Dividing by a large prime number to minimize chance of getting integer values
        self.NoiseZ = self.seed/7919  # Not sure that this is necessary in OpenSimplex but it is so for Perlin Noise
        self.NoisePhase = 0
        self.deltaNoise = 0.01
        self.animSpeed = 10

    def generateLatentNoise(self):
        self.latentSpaceArray = np.zeros((16, 16))
        # Sadly the open simplex library does not offer matrix generation, we have to loop over ours
        for x in range(16):
            for y in range(16):
                self.latentSpaceArray[x, y] = self.noiseGenerator.noise3d(self.NoiseX+x/16, self.NoiseY+y/16, self.NoiseZ + cos(self.NoisePhase))
        self.NoisePhase = self.NoisePhase + self.deltaNoise * self.animSpeed
        if self.NoisePhase > 2*pi:
            self.NoisePhase = 0
            self.NoiseX += self.deltaNoise
            self.NoiseY += self.deltaNoise
        self.latentSpaceArray = np.maximum(np.minimum(self.latentSpaceArray, 1), 0)


    def handleInputs(self):
        # Handle Keyboard input
        key = pg.key.get_pressed()
        if key[pg.K_r]:  # Reset draw board when R is pressed
            self.reset()

    def draw(self):
        self.generateLatentNoise()
        # Prepare draw board to be fed to the tf model
        predictArray = self.latentSpaceArray.reshape((1, 16, 16, 1))

        # Get a reconstruction from the model and reshape it to a 2D matrix
        reconstructPixelArray = (self.model.predict(predictArray).reshape((256, 256)) * 255).astype(np.uint8)

        # Scale the output to 512x512
        reconstructPixelArray = np.repeat(np.repeat(reconstructPixelArray, 2, axis=0), 2, axis=1)

        # Scale the latent space to 512x512
        latentSpacePixelArray = np.repeat(np.repeat(self.latentSpaceArray, 32, axis=0), 32, axis=1) * 255

        # Put the draw board and the reconstruction next to each other
        pixelArray = np.concatenate((latentSpacePixelArray, reconstructPixelArray))

        # 'expand' each pixel from grayscale to rgb (by replicating each pixel 3 times)
        pixelArray = np.dstack((pixelArray, pixelArray, pixelArray))

        # Draw array to screen
        pg.surfarray.blit_array(self.screen, pixelArray)


if __name__ == '__main__':
    animation = animationDisplay()
    animation.runMainLoop()
