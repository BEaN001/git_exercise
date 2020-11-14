import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
# import skimage.io
# import skimage.transform
import os
import matplotlib.pyplot as plt


####### 1.Write TFrecord data

##### 2. Read TFrecord data
def read_and_decode(tfrecords_file, batch_size):
    feature = {'train/person_image': tf.FixedLenFeature([], tf.string),
               'train/gt_heatmap': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return filename and file

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    person_image = tf.decode_raw(features['train/person_image'], tf.float64)
    person_image = tf.reshape(person_image, [128, 128, 3])
    # person_image = tf.image.per_image_standardization(person_image)
    # person_image = person_image / 255.0 - 0.5#normalisze already done in write tfrecord

    gt_heatmap = tf.decode_raw(features['train/gt_heatmap'], tf.float32)
    gt_heatmap = tf.reshape(gt_heatmap, [32, 32, 16])

    # Creates batches by randomly shuffling tensors
    person_image_batch, gt_heatmap_batch = tf.train.shuffle_batch([person_image, gt_heatmap],
                                                                  batch_size=batch_size,
                                                                  capacity=2000,
                                                                  min_after_dequeue=1000)

    return (person_image_batch, gt_heatmap_batch)
    # (bs,376, 376, 3),(bs,46,46,joints_num)


###### 3. Train and save
batch_size = 1  # 32
joints_num = 16
START_LR = 0.0001
iteration = 10  # 100001
tfrecordfile = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_pose_estimation/cpm_tensorflow/pose_train.tfrecords'
data_images_batch, data_gt_heatmap_batch = read_and_decode(tfrecordfile, batch_size)
# (bs,376, 376, 3), (bs,376,376,1),(bs,46,46,joints_num)
data_images = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 3])  # input x1
# data_center_map = tf.placeholder(tf.float32, shape = [batch_size, 376, 376, 1])# input x2
data_gt_heatmap = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, joints_num])  # input y


# CPM FCN
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


output = cpm_small_v2(data_images, joints_num)
loss = tf.nn.l2_loss(data_gt_heatmap - output) / batch_size  # tf.losses.mean_squared_error() tf.nn.l2_loss
loss_5 = tf.stack([tf.nn.l2_loss(data_gt_heatmap - o) for o in tf.get_collection('heatmaps')]) / batch_size

intermediate_loss = tf.reduce_sum(loss_5)
total_loss = loss + intermediate_loss

tf.summary.scalar('stage1_loss', loss_5[0])
tf.summary.scalar('stage2_loss', loss_5[1])
tf.summary.scalar('stage3_loss', loss_5[2])
tf.summary.scalar('stage4_loss', loss_5[3])
tf.summary.scalar('stage5_loss', loss_5[4])
tf.summary.scalar('stage6_loss', loss)
tf.summary.scalar('total_loss', total_loss)  # add loss to scalar summary

# learning rate decay
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(START_LR, global_step, decay_steps=10000, decay_rate=0.98, staircase=False)
tf.summary.scalar('learning_rate', learning_rate)  # add loss to scalar summary
train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

# constant lR
# train_op = tf.train.AdamOptimizer(START_LR).minimize(total_loss)

with tf.Session() as sess:
    saver = tf.train.Saver()  # define a saver for saving and restoring
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logsnew/", sess.graph)  # write to file
    merge_op = tf.summary.merge_all()
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for iter in range(iteration):  # train
        data_images_, data_gt_heatmap_ = sess.run([data_images_batch, data_gt_heatmap_batch])
        # plt.subplot(2,2,1)
        # plt.imshow(data_images_[0,...])
        # plt.imshow(data_gt_heatmap_[0,:,:,0], alpha = 0.5)
        # plt.subplot(2,2,2)
        # plt.imshow(data_images_[0,...])
        # plt.imshow(data_gt_heatmap_[0,:,:,0], alpha = 0.5)
        # plt.subplot(2,2,3)
        # plt.imshow(data_images_[0,...])
        # plt.imshow(data_gt_heatmap_[0,:,:,1], alpha = 0.5)
        # plt.subplot(2,2,4)
        # plt.imshow(data_images_[0,...])
        # plt.imshow(data_gt_heatmap_[0,:,:,2], alpha = 0.5)
        # plt.show()
        #  print(np.max(data_images_),np.min(data_images_),np.max(data_gt_heatmap_),np.min(data_gt_heatmap_))
        # (bs,, 376, 3), (bs,376,376,1),(bs,46,46,joints_num)
        feed_dict = {data_images: data_images_, data_gt_heatmap: data_gt_heatmap_}
        _, total_loss_, output_, l1, l2, loss_5_, result = sess.run(
            [train_op, total_loss, output, loss, intermediate_loss, loss_5, merge_op], feed_dict=feed_dict)
        writer.add_summary(result, iter)

        lr = sess.run(learning_rate)
        if iter % 10 == 0:  # one epoch num_traindata/batch_size
            print('Iter:', iter, '| total_loss: %.4f' % total_loss_, '| intermediate_loss: %.4f' % l2,
                  '| Finalstage_loss: %.4f' % l1, '| learning rate: %.8f' % lr)
            print('5stage_loss:', loss_5_)
            # print('Iter:', iter,'| train loss: %.4f' % total_loss_)
            # plt.subplot(1,2,1)
            # plt.imshow(data_gt_heatmap_[0,:,:,0])
            # plt.subplot(1,2,2)
            # plt.imshow(output_[0,:,:,0])
            # plt.show()
            # print(lr)
        if iter % 100000 == 0:
            saver.save(sess, './modelnew/trained_model_%d.ckpt' % iter)
            # Stop the threads
    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)
    saver.save(sess, './modelnew/trained_model_final.ckpt')

