#tensorflow基于mnist数据集上的VGG11网络，可以直接运行
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import matplotlib.pyplot as plt
#tensorflow基于mnist实现VGG11
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

