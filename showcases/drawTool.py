import numpy as np
import pygame as pg
from tensorflow.keras.models import load_model
from pgUtilities.graphics import drawLine
from pgUtilities.PGWindow import PGWindow
import os
"""
This is the real-time user drawing and reconstruction using PyGame
"""


class drawTool(PGWindow):
    def __init__(self):
        super().__init__()

        # Pixel matrices
        self.imageArray = np.zeros((256, 256))
        self.pixelArray = np.zeros((512, 512 * 2, 3))

        # Load the tensorflow model
        cwd = os.getcwd()
        self.model = load_model(os.path.join(os.path.join(os.path.join(os.path.dirname(cwd)), "models"), "35.h5"))

    def handleEvent(self, event: pg.event) -> None:
        pass

    def handleInputs(self):
        # Handle Keyboard input
        key = pg.key.get_pressed()
        if key[pg.K_r]:  # Reset draw board when R is pressed
            self.imageArray = np.zeros((256, 256))

        # Handle Mouse input
        mouseStatus = pg.mouse.get_pressed()
        rel = pg.mouse.get_rel()
        pos = pg.mouse.get_pos()
        if mouseStatus[0]:
            drawLine(pos, rel, self.imageArray, 255, 4)  # Draw when LMouse is clicked
        if mouseStatus[2]:
            drawLine(pos, rel, self.imageArray, 0, 4)  # Erase when RMouse is clicked

    def draw(self):
        # Prepare draw board to be fed to the tf model
        predictArray = self.imageArray.reshape((1, 256, 256, 1)) / 255

        # Get a reconstruction from the model and reshape it to a 2D matrix
        reconstructPixelArray = (self.model.predict(predictArray).reshape((256, 256)) * 255).astype(np.uint8)

        # Scale the output to 512x512
        reconstructPixelArray = np.repeat(np.repeat(reconstructPixelArray, 2, axis=0), 2, axis=1)

        # Scale the draw board to 512x512, and make the background gray
        drawPixelArray = np.repeat(np.repeat(np.minimum(self.imageArray + 30, 255), 2, axis=0), 2, axis=1)

        # Put the draw board and the reconstruction next to each other
        pixelArray = np.concatenate((drawPixelArray, reconstructPixelArray))

        # 'expand' each pixel from grayscale to rgb (by replicating each pixel 3 times)
        pixelArray = np.dstack((pixelArray, pixelArray, pixelArray))

        # Draw array to screen
        pg.surfarray.blit_array(self.screen, pixelArray)


if __name__ == '__main__':
    drawTool = drawTool()
    drawTool.runMainLoop()
