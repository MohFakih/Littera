import tensorflow as tf

"""
Contains functions used in loading images into memory
"""


def process_img(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img)
    img = tf.image.resize(img, (256, 256))
    img = img[:, :, 3]
    img = tf.expand_dims(img, 2)
    img = img / 255
    return img


class ImagePipeline:
    def __init__(self, letterPath=".\\letters\\*.png", batch_size=32):
        self.letterPath = letterPath
        self.batch_size = batch_size
        filename_ds = tf.data.Dataset.list_files(self.letterPath)
        self.image_ds = filename_ds.map(process_img)
        self.batched_image_ds = self.image_ds.batch(self.batch_size)

    def get_batch(self):
        return next(self.batched_image_ds.take(1).as_numpy_iterator())

    def get_single(self):
        return next(self.image_ds.take(1).as_numpy_iterator())
