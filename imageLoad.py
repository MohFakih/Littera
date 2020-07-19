import tensorflow as tf


class ImagePipeline:
    """
    TensorFlow pipeline for loading images and batching them
    """
    def __init__(self, letterPath=".\\letters\\*.png", batch_size=32, img_size=(256, 256)):
        self.img_size = img_size
        self.letterPath = letterPath
        self.batch_size = batch_size
        filename_ds = tf.data.Dataset.list_files(self.letterPath)
        self.image_ds = filename_ds.map(self.process_img)
        self.batched_image_ds = self.image_ds.batch(self.batch_size)

    def process_img(self, filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_png(img)
        img = tf.image.resize(img, self.img_size)
        img = img[:, :, 3]
        img = tf.expand_dims(img, 2)
        img = img / 255
        return img

    def get_batch(self):
        return next(self.batched_image_ds.take(1).as_numpy_iterator())

    def get_single(self):
        return next(self.image_ds.take(1).as_numpy_iterator())
