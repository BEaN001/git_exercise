import sys
import os
import argparse

import numpy as np
import cv2

from computer_vision.projects.human_detection.mtcnn_tensorflow.tools import IoU

net = 24  # 第二个网络改为24*72

label_file = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/mtcnn_tensorflow/human_detect_stand_one.txt'
im_dir = '/Users/binyu/Downloads/images'

save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24'
pos_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/positive'
part_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/part'
neg_save_dir = '/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/negtive'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)

if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)

if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)

f1 = open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/pos_24.txt', 'w')
f2 = open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/neg_24.txt', 'w')
f3 = open('/Users/binyu/Documents/git_exercise/computer_vision/projects/human_detection/objectdetection_data/native_24/part_24.txt', 'w')

with open(label_file, 'r') as f:
    labels = f.readlines()

print('%d pitures totally' % len(labels))

# for label in labels:
# label =
pos_idx = 0  # positive
neg_idx = 0  # negative
part_idx = 0  # dont care
idx = 0
box_idx = 0
for label in labels:
    label = label.strip().split('\t')
    im_name = label[0]
    bbox = list(map(float, label[1:]))
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    img = cv2.imread(os.path.join(im_dir, im_name))
    idx += 1
    if idx % 100 == 0:
        print('%s images done, pos: %s part: %s neg: %s' %
              (idx, pos_idx, part_idx, neg_idx))

    height, width, channel = img.shape
    #   print(img.shape)

    neg_num = 0

    while neg_num < 40:
        size_x = np.random.randint(24, min(width, height) / 3)
        size_y = 3 * size_x
        nx = np.random.randint(0, width - size_x)
        ny = np.random.randint(0, height - size_y)
        crop_box = np.array([nx, ny, nx + size_x, ny + size_y])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size_y, nx: nx + size_x, :]
        resized_im = cv2.resize(cropped_im, (net, net * 3),
                                interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            save_file = os.path.join(neg_save_dir, '%s.jpg' % neg_idx)
            #         print(save_file)
            f2.write(save_dir + '/negtive/%s' % neg_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            neg_idx += 1
            neg_num += 1
    for box in boxes:

        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        for i in range(5):
            size_x = np.random.randint(24, min(width, height) / 3)
            size_y = size_x * 3
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size_x, -x1), w)
            delta_y = np.random.randint(max(-size_y, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size_x > width or ny1 + size_y > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size_x, ny1 + size_y])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size_y, nx1: nx1 + size_x, :]
            resized_im = cv2.resize(cropped_im, (net, net * 3), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % neg_idx)
                f2.write(save_dir + "/negtive/%s" % neg_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                neg_idx += 1
                # pdb.set_trace()

        for i in range(20):
            size_x = np.random.randint(int(min(w, h) * 0.8),
                                       np.ceil(1.2 * min(w, h)))
            size_y = size_x * 3

            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - size_x / 2, 0))  # nx1 = x1+(-0.3~0.3w) with size_x
            ny1 = int(max(y1 + h / 2 + delta_y - size_y / 2, 0))  # newy1 = y1+(-0.3~0.3h)
            nx2 = nx1 + size_x
            ny2 = ny1 + size_y

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size_x)
            offset_y1 = (y1 - ny1) / float(size_y)
            offset_x2 = (x2 - nx2) / float(size_x)
            offset_y2 = (y2 - ny2) / float(size_y)

            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (net, net * 3),
                                    interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, '%s.jpg' % pos_idx)
                f1.write(save_dir + '/positive/%s' % pos_idx +
                         ' 1 %.2f %.2f %.2f %.2f\n' %
                         (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                pos_idx += 1
            elif IoU(crop_box, box_) >= 0.45:
                save_file = os.path.join(part_save_dir, '%s.jpg' % part_idx)
                f3.write(save_dir + '/part/%s' % part_idx +
                         ' -1 %.2f %.2f %.2f %.2f\n' %
                         (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                part_idx += 1
        box_idx += 1

f1.close()
f2.close()
f3.close()