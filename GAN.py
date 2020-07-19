import tensorflow as tf
import datetime
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, ReLU, LeakyReLU, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import Sequential, Model
from imageLoad import ImagePipeline
import numpy as np
import matplotlib.pyplot as plt


class GAN:
    def __init__(self, batchSize, epochs):

        # Training params
        self.epochs = epochs
        self.batchSize = batchSize
        self.discriminatorOptimizer = Adadelta(1.0)
        self.combinedOptimizer = Adadelta(0.3)

        # Model params
        self.imgDim = (64, 64)
        self.imgTensorDim = (self.imgDim[0], self.imgDim[1], 1)
        self.latentSpaceDim = (16, 1)
        self.discriminatorDropout = 0.3
        self.generatorDropout = 0.3

        # IO params
        self.imagePipeline = ImagePipeline(batch_size=batchSize, img_size=self.imgDim)
        self.saveEvery = 50
        self.savePath = ".\\GANmodels\\"
        self.tensorBoardLogDir = ".\\tensorboardLogs\\"
        # We generate an image from the same tensor to keep track of the training progress visually
        self.referenceLatentTensor = self.sampleLatentTensors(1)

        self.modelsReady = False

    def loadModels(self, modelPath=".\\GANmodels\\19.h5"):
        combinedModel = load_model(modelPath)
        loadedGenerator = combinedModel.layers[1]
        loadedDiscriminator = combinedModel.layers[2]

        self.populateModels()

        self.discriminator.set_weights(loadedDiscriminator.get_weights())
        self.generator.set_weights(loadedGenerator.get_weights())
        self.compileModels()

    def createAndCompileModels(self):
        self.populateModels()
        self.compileModels()

    def populateModels(self):
        # Initialization of the generator, discriminator and the combined models
        # Build Discriminator
        self.discriminator = self.build_discriminator()

        # Build Generator
        self.generator = self.build_generator()
        self.modelsReady = False

    def compileModels(self):
        # Compile discriminator
        self.discriminator.compile(optimizer=self.discriminatorOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # Setup generator
        noise = Input(shape=self.latentSpaceDim)
        generatedImg = self.generator(noise)
        # Combined
        self.discriminator.trainable = False
        decision = self.discriminator(generatedImg)
        self.combinedModel = Model(noise, decision)
        self.combinedModel.compile(optimizer=self.combinedOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.modelsReady = True

    def train(self):
        assert self.modelsReady, "Models have not been compiled yet"
        # Constant labels
        fakeTargetVector = tf.zeros((self.batchSize, 1))
        realTargetVector = tf.ones((self.batchSize, 1))

        # TensorBoard logging initialization
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.tensorBoardLogDir + current_time + '/train'
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Maing training loop
        for epoch in range(self.epochs):
            # sample n images from REAL (dataset) images
            realImages = self.imagePipeline.get_batch()

            # sample n tensors in the latent space and generate the 256x256 images relating to them
            fakeImages = self.generateFakeBatch(self.batchSize)

            # Train discriminator
            fakeLoss = self.discriminator.train_on_batch(fakeImages, fakeTargetVector)
            realLoss = self.discriminator.train_on_batch(realImages, realTargetVector)
            discriminatorLoss = 0.5 * np.add(np.array(fakeLoss), np.array(realLoss))
            # print("Discriminator Training Loss:", discriminatorLoss[0], "\t Discriminator Training Accuracy: ", discriminatorLoss[1])

            # sample n images from REAL (dataset) images
            realImages = self.imagePipeline.get_batch()

            # sample n tensors in the latent space and generate the 256x256 images relating to them
            fakeImages = self.generateFakeBatch(self.batchSize)

            # Test discriminator
            fakeLoss = self.discriminator.evaluate(realImages, realTargetVector, verbose= False)
            realLoss = self.discriminator.evaluate(fakeImages, fakeTargetVector, verbose= False)
            discriminatorLoss = 0.5 * np.add(np.array(fakeLoss), np.array(realLoss))
            print("Discriminator Test Loss:", discriminatorLoss[0], "\t Discriminator Test Accuracy: ", discriminatorLoss[1])


            # Train generator
            latentSpaceNoiseVectors = self.sampleLatentTensors(self.batchSize)
            generatorLoss = self.combinedModel.train_on_batch(latentSpaceNoiseVectors, realTargetVector)
            # print("Generator Training Loss: ", generatorLoss[0], "\t Generator Training Accuracy: ", generatorLoss[1])

            # Test generator
            latentSpaceNoiseVectors = self.sampleLatentTensors(self.batchSize)
            generatorLoss = self.combinedModel.evaluate(latentSpaceNoiseVectors, realTargetVector, verbose= False)
            print("Generator Test Loss: ", generatorLoss[0], "\t Generator Test Accuracy: ", generatorLoss[1])


            # Log test losses for TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('discriminatorLoss', discriminatorLoss[0], step=epoch)
                tf.summary.scalar('generatorLoss', generatorLoss[0], step=epoch)

            # Show generated image
            refImg = self.generateReferenceImage()
            plt.imshow(refImg.reshape(self.imgDim), cmap='gray')
            plt.show()

            if epoch % self.saveEvery == 0:
                self.combinedModel.save(self.savePath+str(epoch//self.saveEvery)+".h5")


    def build_generator(self):
        generator = Sequential([
            Reshape((4, 4, 1)),
            Dropout(self.generatorDropout),
            Conv2DTranspose(32, kernel_size=4, strides=4, padding='same', name='Conv1'),
            BatchNormalization(),
            ReLU(),
            Dropout(self.generatorDropout),
            Conv2DTranspose(16, kernel_size=4, strides=4, padding='same'),
            BatchNormalization(),
            ReLU(),
            Dropout(self.generatorDropout),
            Conv2D(1, kernel_size=4, activation='sigmoid', padding='same')
        ])

        noise = Input(shape=(self.latentSpaceDim[0], self.latentSpaceDim[1]), name='LatentSpaceInput')
        generatedImg = generator(noise)

        return Model(noise, generatedImg)

    def build_discriminator(self):
        discriminator = Sequential([
            Conv2D(32, kernel_size=8, strides=8, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(self.discriminatorDropout),
            Conv2D(8, kernel_size=4, strides=4, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(self.discriminatorDropout),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        inputImg = Input(self.imgTensorDim)
        decision = discriminator(inputImg)

        return Model(inputImg, decision)

    def sampleLatentTensors(self, n):
        return tf.random.normal(shape=(n, self.latentSpaceDim[0], self.latentSpaceDim[1]))

    def generateFakeBatch(self, n):
        latentSpaceNoiseVectors = self.sampleLatentTensors(n)
        generatedImages = self.generator.predict(latentSpaceNoiseVectors)
        return generatedImages

    def generateReferenceImage(self):
        return self.generator.predict(self.referenceLatentTensor)

