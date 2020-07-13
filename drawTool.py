import numpy as np
import pygame as pg
from tensorflow.keras.models import load_model
from graphics import drawLine

"""
This is the real-time user drawing and reconstruction using pygame
"""

# Load the tensorflow model
model = load_model(".\\models\\35.h5")

# Pygame formalities
pg.init()
size = width, height = 512 * 2, 512
screen = pg.display.set_mode(size)
pg.display.set_caption("Littera")
reconstructScreen = pg.display.set_mode(size)

# Flags and initialisation of needed matrices
exitFlag = False
imageArray = np.zeros((256, 256))
pixelArray = np.zeros((512, 512 * 2, 3))

clock = pg.time.Clock()  # Needed to limit framerate

# Main pygame loop
while True:
    clock.tick(30)  # limit FPS to 30

    # Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exitFlag = True
    if exitFlag:
        break

    # Handle Keyboard input
    key = pg.key.get_pressed()
    if key[pg.K_r]:  # Reset draw board when R is pressed
        imageArray = np.zeros((256, 256))

    # Handle Mouse input
    mouseStatus = pg.mouse.get_pressed()
    rel = pg.mouse.get_rel()
    pos = pg.mouse.get_pos()
    if mouseStatus[0]:
        drawLine(pos, rel, imageArray, 255, 4)  # Draw when LMouse is clicked
    if mouseStatus[2]:
        drawLine(pos, rel, imageArray, 0, 4)    # Erase when RMouse is clicked

    # Prepare draw board to be fed to the tf model
    predictArray = imageArray.reshape((1, 256, 256, 1)) / 255

    # Get a reconstruction from the model and reshape it to a 2D matrix
    reconstructPixelArray = (model.predict(predictArray).reshape((256, 256)) * 255).astype(np.uint8)

    # Scale the output to 512x512
    reconstructPixelArray = np.repeat(np.repeat(reconstructPixelArray, 2, axis=0), 2, axis=1)

    # Scale the draw board to 512x512, and make the background gray
    drawPixelArray = np.repeat(np.repeat(np.minimum(imageArray + 30, 255), 2, axis=0), 2, axis=1)

    # Put the draw board and the reconstruction next to each other
    pixelArray = np.concatenate((drawPixelArray, reconstructPixelArray))

    # 'expand' each pixel from grayscale to rgb (by replicating each pixel 3 times)
    pixelArray = np.dstack((pixelArray, pixelArray, pixelArray))

    # Draw to screen
    pg.surfarray.blit_array(screen, pixelArray)
    pg.display.flip()
