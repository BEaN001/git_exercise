import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
from tensorflow.contrib import slim
import tensorflow.contrib.layers as layers


# 2. Read TFrecord data

def read_and_decode_cls(tfrecords_file, batch_size):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'label_raw': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return filename and file

    features = tf.parse_single_example(serialized_example, features=feature)
    # Decode the record read by the reader
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) * (1. / 128.0)
    image.set_shape([72 * 24 * 3])
    image = tf.reshape(image, [72, 24, 3])

    label = tf.decode_raw(features['label_raw'], tf.float32)
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    label.set_shape([2])

    # Creates batches by randomly shuffling tensors
    image_cls_batch, label_cls_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=batch_size,
                                                              capacity=300 + 6 * batch_size,
                                                              min_after_dequeue=10)
    return (image_cls_batch, label_cls_batch)


def read_and_decode_bbx(tfrecords_file, batch_size):
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'label_raw': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return filename and file

    features = tf.parse_single_example(serialized_example, features=feature)
    # Decode the record read by the reader
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) * (1. / 128.0)
    image.set_shape([72 * 24 * 3])
    image = tf.reshape(image, [72, 24, 3])

    label = tf.decode_raw(features['label_raw'], tf.float32)
    label.set_shape([4])

    # Creates batches by randomly shuffling tensors
    image_bbx_batch, label_bbx_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=batch_size,
                                                              capacity=300 + 6 * batch_size,
                                                              min_after_dequeue=10)
    return (image_bbx_batch, label_bbx_batch)


# pdb.set_trace()
###### 3. Train and save
batch_size = 1
# joints_num=16
START_LR = 0.0001
iteration = 2000
tfrecord_files = ['/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/RNet_data_for_cls.tfrecords',
                  '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/RNet_data_for_bbx.tfrecords']
# tfrecordfile = 'E:/2018.6-2018.9_ Disertation/01_My pose estimation/v3/pose_train_ten.tfrecords'
# data_images_batch, data_gt_heatmap_batch = read_and_decode(tfrecordfile, batch_size)

# one for cls/ one for bbx
image_cls_batch, label_cls_batch = read_and_decode_cls(tfrecord_files[0], batch_size)
image_bbx_batch, label_bbx_batch = read_and_decode_bbx(tfrecord_files[1], batch_size)

data_images = tf.placeholder(tf.float32, shape=[batch_size, 72, 24, 3])  # input x1
# data_center_map = tf.placeholder(tf.float32, shape = [batch_size, 376, 376, 1])# input x2
data_label_cls = tf.placeholder(tf.float32, shape=[batch_size, 2])  # output y
data_label_bbx = tf.placeholder(tf.float32, shape=[batch_size, 4])  # output y


# define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def R_Net(inputs):
    with tf.variable_scope('R_Net'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='valid'):
            # pdb.set_trace()
            net = slim.conv2d(inputs, num_outputs=28, kernel_size=[5, 3], stride=1, scope="rconv1")  # 68, 22, 28
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="rpool1", padding='SAME')  # 34, 11, 28
            net = slim.conv2d(net, num_outputs=48, kernel_size=[5, 3], stride=1, scope="rconv2")  # 30, 9, 48
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="rpool2")  # 14, 4, 48)
            net = slim.conv2d(net, num_outputs=64, kernel_size=[5, 3], stride=1, scope="rconv3")  # 10, 2, 64

            net = slim.conv2d(net, num_outputs=128, kernel_size=[5, 1], stride=1, scope="raddconv3")  # 6, 2, 128

            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="rfc1", activation_fn=prelu)

            # fc2_1 = slim.fully_connected(fc1,num_outputs=2,scope="rfc2_1",activation_fn=tf.nn.softmax)
            fc2_1 = slim.fully_connected(fc1, num_outputs=2, scope="rfc2_1", activation_fn=None)

            fc2_2 = slim.fully_connected(fc1, num_outputs=4, scope="rfc2_2", activation_fn=None)
            return (fc2_1, fc2_2)


out_put = R_Net(data_images)
cls_output = tf.reshape(out_put[0], [-1, 2])
bbx_output = tf.reshape(out_put[1], [-1, 4])

softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_label_cls, logits=cls_output))
# softmax_loss = tf.losses.sparse_softmax_cross_entropy(labels=data_label_cls,logits=cls_output)
square_bbx_loss = tf.reduce_mean(tf.squared_difference(bbx_output, data_label_bbx))

# learning rate decay
global_step_cls = tf.Variable(0)
learning_rate_cls = tf.train.exponential_decay(START_LR, global_step_cls, decay_steps=1000, decay_rate=0.98,
                                               staircase=False)
train_cls = tf.train.AdamOptimizer(learning_rate_cls).minimize(softmax_loss, global_step=global_step_cls)
# constant lR
# train_op = tf.train.AdamOptimizer(START_LR).minimize(total_loss)

# learning rate decay
global_step_bbx = tf.Variable(0)
learning_rate_bbx = tf.train.exponential_decay(START_LR, global_step_bbx, decay_steps=1000, decay_rate=0.98,
                                               staircase=False)
train_bbx = tf.train.AdamOptimizer(learning_rate_bbx).minimize(square_bbx_loss, global_step=global_step_bbx)
# constant lR
# train_op = tf.train.AdamOptimizer(START_LR).minimize(total_loss)


with tf.Session() as sess:
    saver = tf.train.Saver()  # define a saver for saving and restoring

    sess.run(tf.global_variables_initializer())
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for iter in range(iteration):  # train
        train_num = np.random.randint(0, 2)

        # data_images_, data_labels_ = sess.run([image_cls_batch, label_cls_batch])
        # feed_dict={data_images:data_images_, data_label_cls:data_labels_}
        # _,softmax_loss_ = sess.run([train_cls, softmax_loss],feed_dict = feed_dict )
        # lr_cls = sess.run(learning_rate_cls)

        # data_images_, data_labels_ = sess.run([image_bbx_batch, label_bbx_batch])
        # feed_dict={data_images:data_images_, data_label_bbx:data_labels_}
        # _,square_bbx_loss_ = sess.run([train_bbx, square_bbx_loss],feed_dict = feed_dict )
        # lr_bbx = sess.run(learning_rate_bbx)

        if train_num == 0:
            data_images_, data_labels_ = sess.run([image_cls_batch, label_cls_batch])
            feed_dict = {data_images: data_images_, data_label_cls: data_labels_}
            _, softmax_loss_ = sess.run([train_cls, softmax_loss], feed_dict=feed_dict)
            lr_cls = sess.run(learning_rate_cls)
        else:
            data_images_, data_labels_ = sess.run([image_bbx_batch, label_bbx_batch])
            feed_dict = {data_images: data_images_, data_label_bbx: data_labels_}
            _, square_bbx_loss_ = sess.run([train_bbx, square_bbx_loss], feed_dict=feed_dict)
            lr_bbx = sess.run(learning_rate_bbx)
        #  if iter % 5 ==0:
        #       print('Iter:',iter,'| softmax_loss: %.4f' % square_bbx_loss_,'| learning rate_cls: %.8f' % lr_bbx)
        if iter > 5 and iter % 5 == 0:  # one epoch num_traindata/batch_size  10epoch~2500
            print('Iter:', iter, '| softmax_loss: %.4f' % softmax_loss_, '| square_bbx_loss: %.4f' % square_bbx_loss_,
                  '| learning rate_cls: %.8f' % lr_cls, '| learning rate_bbx: %.8f' % lr_bbx)

    # Stop the threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)
    saver.save(sess, '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/model/trained_model.ckpt')




