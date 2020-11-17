import sys
import os
import argparse

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.contrib import slim
from computer_vision.projects.human_detection.mtcnn_tensorflow.tools import detect_human, IoU
import pdb
import matplotlib.pyplot as plt
from time import time


def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def P_Net(inputs):  # 36*12
    with tf.variable_scope('P_Net'):
        # define common param
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='valid'):
            # pdb.set_trace()
            net = slim.conv2d(inputs, 10, kernel_size=[5, 3], stride=1, scope='pconv1')  # 33,10,10 111111
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='ppool1')  # 16,5,10
            net = slim.conv2d(net, num_outputs=16, kernel_size=[5, 3], stride=1, scope='conv2')  # 12,3,16 2222

            net = slim.conv2d(net, num_outputs=24, kernel_size=[5, 1], stride=1, scope='paddconv1')  # 8,3,24 3333

            net = slim.conv2d(net, num_outputs=32, kernel_size=[5, 3], stride=1, scope='pconv3')  # 4,1,32  4444

            net = slim.conv2d(net, num_outputs=48, kernel_size=[4, 1], stride=1, scope='paddconv2')  # 1,1,48 555

            conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='pconv4_1',
                                  activation_fn=tf.nn.softmax)
            conv4_2 = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='pconv4_2',
                                  activation_fn=None)
            return (conv4_1, conv4_2)


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
            fc2_1 = slim.fully_connected(fc1, num_outputs=2, scope="rfc2_1", activation_fn=tf.nn.softmax)

            fc2_2 = slim.fully_connected(fc1, num_outputs=4, scope="rfc2_2", activation_fn=None)
            return (fc2_1, fc2_2)


def O_Net(inputs):
    with tf.variable_scope('O_Net'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='valid'):
            # pdb.set_trace()
            net = slim.conv2d(inputs, num_outputs=32, kernel_size=[5, 3], stride=1, scope="oconv1")  # 140, 46, 32
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="opool1", padding='SAME')  # 70, 23, 32  22
            net = slim.conv2d(net, num_outputs=64, kernel_size=[5, 3], stride=1, scope="oconv2")  # 66, 21, 64
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="opool2")  # 32, 10, 64  333
            net = slim.conv2d(net, num_outputs=64, kernel_size=[5, 3], stride=1, scope="oconv3")  # 28, 8, 64
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="opool3", padding='SAME')  # 14, 4, 64 4444

            net = slim.conv2d(net, num_outputs=64, kernel_size=[5, 3], stride=1, scope="oconv4")  # 10, 2, 64
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="opool4", padding='SAME')  # 5, 1, 64  555

            net = slim.conv2d(net, num_outputs=128, kernel_size=[3, 1], stride=1, scope="oconv5")  # 3, 1, 128 6666

            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="ofc1", activation_fn=prelu)  ### 777
            fc2_1 = slim.fully_connected(fc1, num_outputs=2, scope="ofc2_1", activation_fn=tf.nn.softmax)
            fc2_2 = slim.fully_connected(fc1, num_outputs=4, scope="ofc2_2", activation_fn=None)

            return (fc2_1, fc2_2)


def main():
    threshold = [0.8, 0.8, 0.8]
    minsize = 36
    factor = 0.79
    model_file_pnet = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_12/trained_model/trained_model.ckpt'
    model_file_rnet = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/trained_model/trained_model.ckpt'
    model_file_onet = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_48/trained_model/trained_model.ckpt'

    with tf.Session() as sess:
        #  pdb.set_trace()
        image_pnet = tf.placeholder(tf.float32, [None, None, None, 3])
        pnet = P_Net(image_pnet)
        out_tensor_pnet = pnet
        image_rnet = tf.placeholder(tf.float32, [None, 72, 24, 3])
        rnet = R_Net(image_rnet)
        out_tensor_rnet = rnet
        image_onet = tf.placeholder(tf.float32, [None, 144, 48, 3])
        onet = O_Net(image_onet)
        out_tensor_onet = onet

        saver_pnet = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      'P_Net'))
        saver_rnet = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      'R_Net'))
        saver_onet = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      'O_Net'))
        saver_pnet.restore(sess, model_file_pnet)
        saver_rnet.restore(sess, model_file_rnet)
        saver_onet.restore(sess, model_file_onet)

        def pnet_fun(img):
            return sess.run(
                out_tensor_pnet, feed_dict={image_pnet: img})

        def rnet_fun(img):
            return sess.run(
                out_tensor_rnet, feed_dict={image_rnet: img})

        def onet_fun(img):
            return sess.run(
                out_tensor_onet, feed_dict={image_onet: img})

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            # print(frame.shape)
            # start = time()
            # frame = (frame - 127.5)* 0.0078125
            rectangles = detect_human(frame, minsize,
                                      pnet_fun, rnet_fun, onet_fun,
                                      threshold, factor)
            # stop = time()
            # print(str(stop-start) + "秒")
            for i in range(rectangles.shape[0]):
                cv2.rectangle(frame, (int(rectangles[i][0]), int(rectangles[i][1])),
                              (int(rectangles[i][2]), int(rectangles[i][3])), [255, 0, 0], 2)
                cv2.putText(frame, '%.2f' % rectangles[i][4], (int(rectangles[i][0]), int(rectangles[i][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Human Detect', frame)

            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


