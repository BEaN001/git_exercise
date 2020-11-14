import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import skimage.io
import skimage.transform
import cv2
import matplotlib.pyplot as plt
from time import time


def cpm_small_v2(image, joints_num):  # image: bs,128,128,3  output:bs,32,32,joints_num  change a lot
    with tf.variable_scope('CPM'):
        # stage1
        conv1_stage1 = layers.conv2d(image, 64, 5, 1, activation_fn=None, scope='conv1_stage1')
        conv1_stage1 = tf.nn.relu(conv1_stage1)  # 376,376,128 /128,128,64
        pool1_stage1 = layers.max_pool2d(conv1_stage1, 2, 2)  # 187,187,128 /64,64,64
        conv2_stage1 = layers.conv2d(pool1_stage1, 64, 5, 1, activation_fn=None, scope='conv2_stage1')
        conv2_stage1 = tf.nn.relu(conv2_stage1)  # 187,187,128 /64,64,64
        pool2_stage1 = layers.max_pool2d(conv2_stage1, 2, 2)  # 93,93,128 /32,32,64
        conv4_stage1 = layers.conv2d(pool2_stage1, 16, 3, 1, activation_fn=None, scope='conv4_stage1')
        conv4_stage1 = tf.nn.relu(conv4_stage1)  # 46,46,32 /32,32,16
        conv5_stage1 = layers.conv2d(conv4_stage1, 352, 7, 1, activation_fn=None, scope='conv5_stage1')
        conv5_stage1 = tf.nn.relu(conv5_stage1)  # 46,46,512 /32,32,352
        conv6_stage1 = layers.conv2d(conv5_stage1, 352, 1, 1, activation_fn=None, scope='conv6_stage1')
        conv6_stage1 = tf.nn.relu(conv6_stage1)  # 46,46,512 /32,32,352
        conv7_stage1 = layers.conv2d(conv6_stage1, joints_num, 1, 1, activation_fn=None,
                                     scope='conv7_stage1')  # 46,46,joints_num / 32,32,joints_num
        tf.add_to_collection('heatmaps', conv7_stage1)
        # stage2
        conv1_stage2 = layers.conv2d(image, 64, 5, 1, activation_fn=None, scope='conv1_stage2')
        conv1_stage2 = tf.nn.relu(conv1_stage2)  # 376,376,128 /128,128,64
        pool1_stage2 = layers.max_pool2d(conv1_stage2, 2, 2)  # 187,187,128 /64,64,64
        conv2_stage2 = layers.conv2d(pool1_stage2, 64, 5, 1, activation_fn=None, scope='conv2_stage2')
        conv2_stage2 = tf.nn.relu(conv2_stage2)  # 187,187,128/64,64,64
        pool2_stage2 = layers.max_pool2d(conv2_stage2, 2, 2)  # 93,93,128 /32,32,64
        conv4_stage2 = layers.conv2d(pool2_stage2, 16, 3, 1, activation_fn=None, scope='conv4_stage2')
        conv4_stage2 = tf.nn.relu(conv4_stage2)  # 46,46,32 /32,32,16
        concat_stage2 = tf.concat(axis=3, values=[conv4_stage2, conv7_stage1])  # 46,46,47 / 32,32,32
        Mconv1_stage2 = layers.conv2d(concat_stage2, 64, 5, 1, activation_fn=None, scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)  # 46,46,128 / 32,32,64
        Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 64, 5, 1, activation_fn=None, scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)  # 46,46,128 / 32,32,64
        Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 64, 5, 1, activation_fn=None, scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)  # 46,46,128 / 32,32,64
        Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 64, 1, 1, activation_fn=None, scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)  # 46,46,128 / 32,32,64
        Mconv5_stage2 = layers.conv2d(Mconv4_stage2, joints_num, 1, 1, activation_fn=None,
                                      scope='Mconv5_stage2')  # 46,46,joints_num / 32,32,joints_num
        tf.add_to_collection('heatmaps', Mconv5_stage2)
        # stage 3
        conv1_stage3 = layers.conv2d(pool2_stage2, 16, 3, 1, activation_fn=None, scope='conv1_stage3')
        conv1_stage3 = tf.nn.relu(conv1_stage3)  # 46,46,32
        concat_stage3 = tf.concat(axis=3, values=[conv1_stage3, Mconv5_stage2])  # 46,46,47
        Mconv1_stage3 = layers.conv2d(concat_stage3, 64, 5, 1, activation_fn=None, scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)  # 46,46,128
        Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 64, 5, 1, activation_fn=None, scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)  # 46,46,128
        Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 64, 5, 1, activation_fn=None, scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)  # 46,46,128
        Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 64, 1, 1, activation_fn=None, scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)  # 46,46,128
        Mconv5_stage3 = layers.conv2d(Mconv4_stage3, joints_num, 1, 1, activation_fn=None,
                                      scope='Mconv5_stage3')  # 46,46,joints_num
        tf.add_to_collection('heatmaps', Mconv5_stage3)
        # stage 4
        conv1_stage4 = layers.conv2d(pool2_stage2, 16, 3, 1, activation_fn=None, scope='conv1_stage4')
        conv1_stage4 = tf.nn.relu(conv1_stage4)  # 46,46,32
        concat_stage4 = tf.concat(axis=3, values=[conv1_stage4, Mconv5_stage3])
        Mconv1_stage4 = layers.conv2d(concat_stage4, 64, 5, 1, activation_fn=None, scope='Mconv1_stage4')
        Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)  # 46,46,128
        Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 64, 5, 1, activation_fn=None, scope='Mconv2_stage4')
        Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)  # 46,46,128
        Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 64, 5, 1, activation_fn=None, scope='Mconv3_stage4')
        Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)  # 46,46,128
        Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 64, 1, 1, activation_fn=None, scope='Mconv4_stage4')
        Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)  # 46,46,128
        Mconv5_stage4 = layers.conv2d(Mconv4_stage4, joints_num, 1, 1, activation_fn=None,
                                      scope='Mconv5_stage4')  # 46,46,joints_num
        tf.add_to_collection('heatmaps', Mconv5_stage4)
        # stage 5
        conv1_stage5 = layers.conv2d(pool2_stage2, 16, 3, 1, activation_fn=None, scope='conv1_stage5')
        conv1_stage5 = tf.nn.relu(conv1_stage5)
        concat_stage5 = tf.concat(axis=3, values=[conv1_stage5, Mconv5_stage4])
        Mconv1_stage5 = layers.conv2d(concat_stage5, 64, 5, 1, activation_fn=None, scope='Mconv1_stage5')
        Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
        Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 64, 5, 1, activation_fn=None, scope='Mconv2_stage5')
        Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
        Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 64, 5, 1, activation_fn=None, scope='Mconv3_stage5')
        Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
        Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 64, 1, 1, activation_fn=None, scope='Mconv4_stage5')
        Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
        Mconv5_stage5 = layers.conv2d(Mconv4_stage5, joints_num, 1, 1, activation_fn=None,
                                      scope='Mconv5_stage5')  # 46,46,joints_num
        tf.add_to_collection('heatmaps', Mconv5_stage5)
        # stage 6
        conv1_stage6 = layers.conv2d(pool2_stage2, 16, 3, 1, activation_fn=None, scope='conv1_stage6')
        conv1_stage6 = tf.nn.relu(conv1_stage6)
        concat_stage6 = tf.concat(axis=3, values=[conv1_stage6, Mconv5_stage5])
        Mconv1_stage6 = layers.conv2d(concat_stage6, 64, 5, 1, activation_fn=None, scope='Mconv1_stage6')
        Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
        Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 64, 5, 1, activation_fn=None, scope='Mconv2_stage6')
        Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
        Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 64, 5, 1, activation_fn=None, scope='Mconv3_stage6')
        Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
        Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 64, 1, 1, activation_fn=None, scope='Mconv4_stage6')
        Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
        Mconv5_stage6 = layers.conv2d(Mconv4_stage6, joints_num, 1, 1, activation_fn=None,
                                      scope='Mconv5_stage6')  # 46,46,joints_num
        return Mconv5_stage6  # 46,46,joints_num


def draw_limbs(image, parts):
    # for oid in range(parts.shape[0]):
    for lid, (p0, p1) in enumerate(LIMBS):
        # print(oid,lid,p0,p1)
        # print(parts[0][0],parts[0][1])
        # print(parts[0],parts[1])
        x0, y0 = parts[p0]
        x1, y1 = parts[p1]
        # y0, x0 = parts[oid][p0]
        # y1, x1 = parts[oid][p1]
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[lid], 2)
        cv2.circle(image, (int(x0), int(y0)), 5, COLORS[lid], -1)
        cv2.circle(image, (int(x1), int(y1)), 5, COLORS[lid], -1)


model_path = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_pose_estimation/cpm_tensorflow/trained_model/trained_model.ckpt'

joints_num = 16
LIMBS = np.array(
    [1, 2, 2, 3, 4, 5, 5, 6, 3, 7, 4, 7, 11, 12, 12, 13, 8, 13, 14, 15, 15, 16, 8, 14, 7, 8, 8, 9, 9, 10]).reshape(
    (-1, 2)) - 1
COLORS = [[0, 0, 255], [0, 0, 255], [0, 170, 255], [0, 170, 255], [0, 0, 255], [0, 170, 255],
          [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 170], [255, 0, 170], [255, 0, 170], [0, 255, 0], [0, 255, 0],
          [0, 255, 0]]

with tf.Session() as sess:
    input_image = tf.placeholder(tf.float32, shape=[1, 128, 128, 3])
    output = cpm_small_v2(input_image, joints_num)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             'CPM'))
    saver.restore(sess, model_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        start = time()
        # cv2.imshow('Human_pose',frame)
        # print(frame.shape) #480,640,3
        x_min, y_min, x_max, y_max = 200, 40, 440, 440
        image = frame[y_min:y_max, x_min:x_max, :]
        h, w, _ = image.shape
        image = skimage.transform.resize(image, [128, 128])  # 0~1
        image = image[np.newaxis] - 0.5  # float64 -0.5~0.5

        heatmaps = sess.run(output, {input_image: image})

        joints = np.zeros((joints_num, 2))
        for i in range(joints_num):
            heatmap = heatmaps[0, :, :, i]
            joint = np.where(heatmap == np.max(heatmap))
            x_ori = joint[1] / 32 * w + x_min
            y_ori = joint[0] / 32 * h + y_min
            joints[i, :] = [np.round(x_ori), np.round(y_ori)]
        draw_limbs(frame, joints)

        cv2.imshow('Human_pose', frame)

        # print("Stop: " + str(stop))

        k = cv2.waitKey(10) & 0xFF
        stop = time()
        print(str(stop - start) + "秒")
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()