import os
import numpy as np
from PIL import Image
import tensorflow as tf

"""
Contains functions used in loading images into memory
"""



# TODO: change this system to use tf.data.dataset and/or keras preprocessing, they probably have better data pipelining



def getLanguageDirPaths():
    """
    :return: List of paths for the language folders (each language folder presumably contains sorted letter folders).
    """
    cyrillicPath = ".\letters\Cyrillic"
    latinPath = ".\letters\Latin"
    pathList = [latinPath, cyrillicPath]
    return pathList

def getImagesFromLanguageDir(languagePath):
    """
    :param languagePath: OS path to language dir. Dir should contain sub-directories for each letter images.
    :return: list of paths for all the images in the sub-directories
    """
    imgPathList = []  # Holds path for each individual image
    languageDirList = os.listdir(languagePath)  # List all letter directories
    for letterDir in languageDirList:
        pathToLetterDir = os.path.join(languagePath, letterDir)
        imageNameList = os.listdir(pathToLetterDir)
        imgPathList += [os.path.join(pathToLetterDir, imageName) for imageName in imageNameList]
    return imgPathList


def getPathToAllImages():
    """
    :return: List of paths to each individual image for all languages.
    """
    pathToAllImages = []
    for langPath in getLanguageDirPaths():
        pathToAllImages += getImagesFromLanguageDir(langPath)
    return pathToAllImages

def sampleImages(listOfPaths, n=512, dim1=None, dim2=None, exclude=None):
    """
    :param listOfPaths: List of paths to all the images to sample from
    :param n: Number of images to sample
    :param exclude: Number of images to exclude from the sampling (presumably to keep a validation set)
    :param dim1: dim1 and dim2 are the dimensions of the output images, a None value will not resize the images
    :param dim2: dim1 and dim2 are the dimensions of the output images, a None value will not resize the images
    :return: (n, dim1, dim2, 1) numpy array holding n [dim1 * dim2] grayscale images
    """
    if not exclude:
        pathsOfImagesToLoad = np.random.choice(listOfPaths, n, replace=False)
    else:
        pathsOfImagesToLoad = np.random.choice(listOfPaths[:-exclude], n, replace=False)
    o = []
    for path in pathsOfImagesToLoad:
        img = np.array(Image.open(path))
        dim_x = dim1 if dim1 else img.shape[0]
        dim_y = dim2 if dim2 else img.shape[1]
        img = img[:dim_x, :dim_y, 3].reshape((dim_x, dim_y, 1))
        o.append(img / 255)
    return np.array(o)
