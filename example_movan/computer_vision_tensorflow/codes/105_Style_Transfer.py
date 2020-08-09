"""
An simple implementation of [Image style transferring using CNN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
The following code is based on this [blog](https://harishnarayanan.org/writing/artistic-style-transfer/)
Dependencies:
tensorflow=1.8.0
PIL=5.1.0
scipy=1.1.0
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b

# image and model path
CONTENT_PATH = '/Users/binyu/Desktop/nw.jpeg'
STYLE_PATH = '/Users/binyu/Desktop/stype.jpeg'
OUTPUT_DIR = '/Users/binyu/Desktop/'
VGG_PATH = '/Users/binyu/Downloads/vgg16.npy'

# weight for loss (content loss, style loss and total variation loss)
W_CONTENT = 0.001
W_STYLE = W_CONTENT * 1e3
W_VARIATION = 1.
HEIGHT, WIDTH = 400, 400  # output image height and width
N_ITER = 6  # styling how many times?


class StyleTransfer:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy, w_content, w_style, w_variation, height, width):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy, encoding='latin1', allow_pickle=True).item()
        except FileNotFoundError:
            print(
                'Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.height, self.width = height, width

        # network input (combined images)
        self.tf_content = tf.placeholder(tf.float32, [1, height, width, 3])
        self.tf_style = tf.placeholder(tf.float32, [1, height, width, 3])
        self.tf_styled = tf.placeholder(tf.float32, [1, height, width, 3])
        concat_image = tf.concat((self.tf_content, self.tf_style, self.tf_styled), axis=0)  # combined input

        # convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=concat_image)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG conv layers
        conv1_1 = self._conv_layer(bgr, "conv1_1")
        conv1_2 = self._conv_layer(conv1_1, "conv1_2")
        pool1 = self._max_pool(conv1_2, 'pool1')
        conv2_1 = self._conv_layer(pool1, "conv2_1")
        conv2_2 = self._conv_layer(conv2_1, "conv2_2")
        pool2 = self._max_pool(conv2_2, 'pool2')
        conv3_1 = self._conv_layer(pool2, "conv3_1")
        conv3_2 = self._conv_layer(conv3_1, "conv3_2")
        conv3_3 = self._conv_layer(conv3_2, "conv3_3")
        pool3 = self._max_pool(conv3_3, 'pool3')
        conv4_1 = self._conv_layer(pool3, "conv4_1")
        conv4_2 = self._conv_layer(conv4_1, "conv4_2")
        conv4_3 = self._conv_layer(conv4_2, "conv4_3")
        pool4 = self._max_pool(conv4_3, 'pool4')
        conv5_1 = self._conv_layer(pool4, "conv5_1")
        conv5_2 = self._conv_layer(conv5_1, "conv5_2")
        conv5_3 = self._conv_layer(conv5_2, "conv5_3")

        # we don't need fully connected layers for style transfer

        with tf.variable_scope('content_loss'):  # compute content loss
            content_feature_maps = conv2_2[0]
            styled_feature_maps = conv2_2[2]
            loss = w_content * tf.reduce_sum(tf.square(content_feature_maps - styled_feature_maps))

        with tf.variable_scope('style_loss'):  # compute style loss
            conv_layers = [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]
            for conv_layer in conv_layers:
                style_feature_maps = conv_layer[1]
                styled_feature_maps = conv_layer[2]
                style_loss = (w_style / len(conv_layers)) * self._style_loss(style_feature_maps, styled_feature_maps)
                loss = tf.add(loss, style_loss)  # combine losses

        with tf.variable_scope('variation_loss'):  # total variation loss, reduce noise
            a = tf.square(self.tf_styled[:, :height - 1, :width - 1, :] - self.tf_styled[:, 1:, :width - 1, :])
            b = tf.square(self.tf_styled[:, :height - 1, :width - 1, :] - self.tf_styled[:, :height - 1, 1:, :])
            variation_loss = w_variation * tf.reduce_sum(tf.pow(a + b, 1.25))
            self.loss = tf.add(loss, variation_loss)

        # styled image's gradient
        self.grads = tf.gradients(loss, self.tf_styled)

        self.sess = tf.Session()
        tf.summary.FileWriter('./log', self.sess.graph)

    def styling(self, content_image, style_image, n_iter):
        content = Image.open(content_image).resize((self.width, self.height))
        self.content = np.expand_dims(content, axis=0).astype(np.float32)  # [1, height, width, 3]
        style = Image.open(style_image).resize((self.width, self.height))
        self.style = np.expand_dims(style, axis=0).astype(np.float32)  # [1, height, width, 3]

        x = np.copy(self.content)  # initialize styled image from content

        # repeat backpropagating to styled image
        for i in range(n_iter):
            x, min_val, info = fmin_l_bfgs_b(self._get_loss, x.flatten(), fprime=lambda x: self.flat_grads, maxfun=20)
            x = x.clip(0., 255.)
            print('(%i/%i) loss: %.1f' % (i + 1, n_iter, min_val))

        x = x.reshape((self.height, self.width, 3))
        for i in range(1, 4):
            x[:, :, -i] += self.vgg_mean[i - 1]
        return x, self.content, self.style

    def _get_loss(self, x):
        loss, grads = self.sess.run(
            [self.loss, self.grads], feed_dict={
                self.tf_styled: x.reshape((1, self.height, self.width, 3)),
                self.tf_content: self.content,
                self.tf_style: self.style
            })
        self.flat_grads = grads[0].flatten().astype(np.float64)
        return loss

    def _style_loss(self, style_feature, styled_feature):
        def gram_matrix(x):
            num_channels = int(x.get_shape()[-1])
            matrix = tf.reshape(x, shape=[-1, num_channels])
            gram = tf.matmul(tf.transpose(matrix), matrix)
            return gram

        s = gram_matrix(style_feature)
        t = gram_matrix(styled_feature)
        channels = 3
        size = self.width * self.height
        return tf.reduce_sum(tf.square(s - t)) / (4. * (channels ** 2) * (size ** 2))

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name):  # in here, CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout


if __name__ == '__main__':
    image_filter = StyleTransfer(VGG_PATH, W_CONTENT, W_STYLE, W_VARIATION, HEIGHT, WIDTH, )
    image, content_image, style_image = image_filter.styling(CONTENT_PATH, STYLE_PATH, N_ITER)  # style transfer

    # save
    image = image.clip(0, 255).astype(np.uint8)
    save_name = '_'.join([path.split('/')[-1].split('.')[0] for path in [CONTENT_PATH, STYLE_PATH]]) + '.jpeg'
    Image.fromarray(image).save(''.join([OUTPUT_DIR, save_name]))  # save result