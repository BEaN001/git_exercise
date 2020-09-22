import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np


class ImageData:
    def __init__(self, txtfile, batch_size, num_classes, image_size, buffer_scale=100):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.txt_file = txtfile
        buffer_size = batch_size * buffer_scale
        self.img_path = []
        self.labels = []

        # 读取图片
        self.read_txt_file()
        self.dataset_size = len(self.labels)
        print("num of train datas=", self.dataset_size)
        # 转换成tensor
        self.img_paths = convert_to_tensor(self.img_path, dtype=dtypes.string, name="image_paths")
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32, name="labels")
        # 创建数据集
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        print(type(data))
        data = data.map(self._parse_funciton)
        data = data.repeat(1000)
        data = data.shuffle(buffer_size=buffer_size)

        # 设置self data Batch
        self.data = data.batch(batch_size)
        print("self.data type=", type(self.data))

    def read_txt_file(self):
       for line in open(self.txt_file, 'r'):
           items = line.split(' ')
           self.img_path.append(items[0])
           self.labels.append(int(items[1].split('\n')[0]))

    def augment_dataset(self, image, size):
        distorted_image = tf.image.random_brightness(image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
        return float_image

    def _parse_funciton(self, filename, label):
        label_ = tf.one_hot(label, self.num_classes)
        img = tf.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.random_crop(img, [self.image_size[0], self.image_size[1], 3])
        img = tf.image.random_flip_left_right(img)
        img = self.augment_dataset(img, self.image_size)
        return img, label_


if __name__ == "__main__":

    txtfile = '/Users/binyu/Documents/git_exercise/computer_vision/projects/classfication/simpleconv3/train_shuffle.txt'
    batch_size = 64
    num_classes = 2
    image_size = (48, 48)

    dataset = ImageData(txtfile, batch_size, num_classes, image_size)
    iterator = dataset.data.make_one_shot_iterator()
    dataset_size = dataset.dataset_size
    batch_images, batch_labels = iterator.get_next()