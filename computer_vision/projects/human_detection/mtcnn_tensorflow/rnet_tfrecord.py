import sys
import random

import cv2
import tensorflow as tf
import numpy as np

from computer_vision.projects.human_detection.mtcnn_tensorflow.tools import bytes_feature

pos_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/positive'
part_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/part'
neg_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/negtive'


def main():
    # size = 24
    # net = str(size)
    with open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/pos_24.txt', 'r') as f:
        pos = f.readlines()
    with open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/neg_24.txt', 'r') as f:
        neg = f.readlines()
    with open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/part_24.txt', 'r') as f:
        part = f.readlines()

    ######postive for classification
    print('Writing')
    print('\n' + 'postive for classification')
    filename_cls = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/RNet_data_for_cls.tfrecords'

    examples = []
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename_cls)
    cur_ = 0
    sum_ = len(pos)

    for line in pos:
        cur_ += 1
        if cur_ % 500 == 0:
            print('Train data(postive for classification): {}/{}'.format(cur_, sum_))
        words = line.split()
        # image_file_name = words[0] + ' ' + words[1] + '.jpg'
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        # im = cv2.imread(os.path.join(pos_save_dir, str(cur_-1) + '.jpg'))

        h, w, ch = im.shape
        if h != 72 or w != 24:
            im = cv2.resize(im, (24, 72))
        im = im.astype('uint8')
        label = np.array([0, 1], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)

    ######negtive for classification
    print('\n' + 'negtive for classification ')
    cur_ = 0
    # neg_keep = np.random.choice(len(neg), size=3*len(pos), replace=False)
    sum_ = len(neg)

    for line in neg:
        # line = neg[i]
        cur_ += 1
        if cur_ % 500 == 0:
            print('Train data(negtive for classification): {}/{}'.format(cur_, sum_))
        words = line.split()
        # image_file_name = words[0] + ' ' + words[1] + '.jpg'
        image_file_name = words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != 72 or w != 24:
            im = cv2.resize(im, (24, 72))
        im = im.astype('uint8')
        label = np.array([1, 0], dtype='float32')
        label_raw = label.tostring()
        image_raw = im.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label_raw': bytes_feature(label_raw),
            'image_raw': bytes_feature(image_raw)}))
        examples.append(example)
    print(len(examples))
    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())  ###tfrecord file for classfition done!
    writer.close()
    #######postive random for regression
    examples=[]
    print('Writing')
    print('\n' + 'positive random cropped for regression')
    cur_ = 0
    filename_roi = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/RNet_data_for_bbx.tfrecords'

    writer = tf.python_io.TFRecordWriter(filename_roi)
    # pos_keep = np.random.choice(len(pos), size=len(pos), replace=False)
    sum_ = len(pos)
    print('Writing')
    for line in pos:
    # view_bar(cur_+1, sum_)
        cur_ += 1
    if cur_ % 500 == 0:
        print('Train data(positive random for regression): {}/{}'.format(cur_, sum_))
    # line = pos[i]
    words = line.split()
    # image_file_name = words[0] + ' ' + words[1] + '.jpg'
    image_file_name = words[0] + '.jpg'
    im = cv2.imread(image_file_name)
    h, w, ch = im.shape
    if h != 72 or w != 24:
        im = cv2.resize(im, (24, 72))
    im = im.astype('uint8')
    label = np.array([float(words[2]), float(words[3]),
                      float(words[4]), float(words[5])],
                     dtype='float32')
    label_raw = label.tostring()
    image_raw = im.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': bytes_feature(label_raw),
        'image_raw': bytes_feature(image_raw)}))
    examples.append(example)
    print(len(examples))

    #######part random for regression
    print('\n' + 'part random cropped for regression')
    cur_ = 0
    # part_keep = np.random.choice(len(part), size=len(pos), replace=False)
    sum_ = len(part)
    for line in part:
    # for line in part:
    # line = part[i]
        cur_ += 1
    if cur_ % 500 == 0:
        print('Train data(part random for regression): {}/{}'.format(cur_, sum_))
    words = line.split()
    # image_file_name = words[0] + ' ' + words[1] + '.jpg'
    image_file_name = words[0] + '.jpg'
    im = cv2.imread(image_file_name)
    h, w, ch = im.shape
    if h != 72 or w != 24:
        im = cv2.resize(im, (24, 72))
    im = im.astype('uint8')
    label = np.array([float(words[2]), float(words[3]),
                      float(words[4]), float(words[5])],
                     dtype='float32')
    label_raw = label.tostring()
    image_raw = im.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': bytes_feature(label_raw),
        'image_raw': bytes_feature(image_raw)}))
    examples.append(example)
    print(len(examples))

    random.shuffle(examples)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
