import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform
import sys
import matplotlib.pyplot as plt
# from view import view
import pdb


def data_read(file):  # read  the data.txt  and return addrs, points_min, points_max, joints (list)
    addrs = []
    points_min = []
    points_max = []
    joints = []
    with open(file, 'r') as op:
        for line in op.readlines():
            addr, label = line.split('\t')
            addrs.append(addr)
            label = [float(label) for label in ' '.join(label.split()).split(' ')]
            newlabel = list(zip(*([iter(label)] * 2)))
            point_min, point_max = newlabel[:2]
            points_min.append([point_min])
            points_max.append([point_max])
            joint = newlabel[2:]
            joints.append(joint)
    return addrs, points_min, points_max, joints


def Gaussian_heatmap(img_height, img_width, points, sigma):
    x = np.arange(0, img_width, 1, np.float32)
    y = np.arange(0, img_height, 1, np.float32)[:, np.newaxis]
    channel = len(points)
    gaussian_heatmap = np.zeros([img_height, img_width, channel], dtype=np.float32)
    for c in range(channel):
        x0, y0 = points[c]
        dist_sq = (x - x0) ** 2 + (y - y0) ** 2
        exponent = dist_sq / 2.0 / sigma / sigma
        gaussian_heatmap[:, :, c] = np.exp(-(exponent))
        # for x_p in range(img_width):
        # for y_p in range(img_height):
        # dist_sq = (x_p - x0) **2 +(y_p - y0) **2
        # exponent = dist_sq / 2.0 / sigma / sigma
        # gaussian_heatmap[y_p,x_p,c] = np.exp(-exponent)
    return gaussian_heatmap


##convert data to features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


addrs, points_min, points_max, joints = data_read(
    '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_pose_estimation/cpm_tensorflow/train_data_one.txt')
train_filename = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_pose_estimation/cpm_tensorflow/pose_train.tfrecords'

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(addrs)):
    # print how many images are saved every 1000 images
    if not i % 100:
        print('Train data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()
    addr = addrs[i]
    point_min = points_min[i]  # [(x_min,y_min)]
    point_max = points_max[i]  # [(x_max,y_max)]
    joints_16 = joints[i]
    # print(point_min,point_min[0][1])
    # print(point_max,point_max[0][1])
    # print(joints_16,joints_16[0])
    # load the image
    # H, W =376, 376 # input size for cpm
    H, W = 128, 128
    image = skimage.io.imread(addr)  # RGB find the human pose ->376,376,3
    # plt.imshow(image),plt.show()
    # print(image.shape,image.dtype,np.max(image))
    # image = image/255.0-0.5
    # person_image = np.zeros((int(point_max[0][1])-int(point_min[0][1]), int(point_max[0][0])-int(point_min[0][0]),3))  #do not need
    # person_image = np.zeros((int(point_max[0][1])-int(point_min[0][1])+1, int(point_max[0][0])-int(point_min[0][0])+1,3))
    person_image = image[int(point_min[0][1]):int(point_max[0][1]), int(point_min[0][0]):int(point_max[0][0]), :]
    # plt.imshow(person_image),plt.show()
    h, w, _ = person_image.shape
    # print(np.max(person_image),person_image.dtype)#unit 8  0~255
    # print(person_image.shape)
    # person_image_resize = tf.image.resize_images(person_image, [H,W])    #h, w, 3->376,376,3
    # since I can not change a tensor to byte, so just convert person_image to features
    person_image_resize = skimage.transform.resize(person_image, [H, W]) - 0.5  # float64 -0.5~0.5
    # plt.imshow(person_image_resize),plt.show()
    #   person_image_resize = (person_image_resize*255).astype(np.uint8) #uint8 0~255
    # person_image_resize = (person_image_resize*255).astype(np.uint8)
    # print(np.max(person_image_resize),person_image_resize.dtype) #float64 0~1
    # print((person_image_resize*255).astype(np.uint8))
    #  person_image_resize = person_image_resize/255.0 -0.5 #normalise -0.5~0.5 float64
    # print(np.max(person_image_resize),person_image_resize.dtype) #float64 -0.5~0.5
    # plt.imshow(person_image_resize)
    # plt.show()
    img_raw = person_image_resize.tobytes()

    joints_16_box = []
    # print(point_min[0][0])
    # print(point_min[0][1])
    # print(joints_16)
    for i in range(len(joints_16)):
        x, y = joints_16[i]
        x = x - point_min[0][0]  # V3 x-=point_min[0][0]
        y = y - point_min[0][1]  # V3 y-=point_min[0][1]
        joints_16_box.append((x, y))
    # print(joints_16_box)

    # print(h,w)
    joints_16_resize = []  # h, w-> 46,46
    for i in range(len(joints_16_box)):
        x, y = joints_16_box[i]
        x = x / w * 32  # V2 int(x/w*46) =>x/w*46
        y = y / h * 32  # V2 int(y/h*46) =>y/h*46
        joints_16_resize.append((x, y))
    # print(joints_16_resize)
    # print(len(joints_16_resize))
    # plt.imshow(person_image_resize)
    # plt.show()

    # center_map = Gaussian_heatmap()
    sigma = 1.2  # sigma ?
    # pdb.set_trace()
    gt_heatmap = Gaussian_heatmap(32, 32, joints_16_resize, sigma)

    # allheatmap=np.zeros((46,46))
    # for i in range(16):
    # allheatmap=allheatmap+gt_heatmap[:,:,i]#heatmap for keypoint
    # plt.imshow(allheatmap),plt.show()
    # view(person_image_resize, gt_heatmap, show_max=False)
    # print(gt_heatmap.dtype,np.max(gt_heatmap),np.min(gt_heatmap))#float 32, 1, 0
    gt_heatmap_raw = gt_heatmap.tobytes()
    # plt.imshow(gt_heatmap[:,:,1])
    # plt.show()
    # plt.subplot(2,2,1)
    # plt.imshow(person_image_resize)
    # plt.subplot(2,2,2)
    # plt.imshow(gt_heatmap[:,:,0])
    # plt.subplot(2,2,3)
    # plt.imshow(gt_heatmap[:,:,1])
    # plt.subplot(2,2,4)
    # plt.imshow(gt_heatmap[:,:,2])
    # plt.show()

    # Create a feature
    feature = {'train/person_image': _bytes_feature(img_raw),
               'train/gt_heatmap': _bytes_feature(gt_heatmap_raw),
               }

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

