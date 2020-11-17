import os
import sys
import re

import numpy as np
import tensorflow as tf
import cv2
import pdb
import matplotlib.pyplot as plt
from time import time


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def detect_human(img, minsize, pnet, rnet, onet, threshold, factor):
    factor_count = 0
    h = img.shape[0]
    w = img.shape[1]

    # img_h, img_w, _ = img.shape
    # min_size = min(img_h, img_w)
    # scales, scale = [], 1.0
    # total_boxes = np.empty((0, 9))
    # while min_size >= minsize:

    # scales.append(scale)
    ##factor:
    # min_size *= factor
    # scale *= factor

    img_h, img_w, _ = img.shape
    # min_size = min(img_h, img_w)
    scales, scale = [], 1.0
    total_boxes = np.empty((0, 9))
    while img_h >= 36 and img_w >= 12:
        #
        scales.append(scale)
        # factor:
        img_h *= factor
        img_w *= factor
        scale *= factor
        # scales = [0.79*0.79*0.79]
    # first stage
    # start1 = time()
    for j in range(len(scales)):
        scale = scales[j]

        # start = time()
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        # stop = time()
        # print(str(stop-start) + "second for part1_1")
        # start = time()
        im_data = imresample(img, (hs, ws))
        # im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        # stop = time()
        # print(str(stop-start) + "second for part1_2")
        # pdb.set_trace()
        # start = time()
        # im_data = (im_data - 127.5) * (1. / 128.0)
        im_data = (im_data - 127.5) * 0.0078125  ############ NORMALISZE!
        # stop = time()
        # print(str(stop-start) + "second for part1_3")
        # start = time()
        img_x = np.expand_dims(im_data, 0)
        # stop = time()
        # print(str(stop-start) + "second for part1_4")

        # start = time()
        out = pnet(img_x)  ##############PPPPPPPPPPPPPPPPPPPP
        # stop = time()
        # print(str(stop-start) + "second for PNet run")
        # start = time()
        out0 = out[0]  # (1,410,302,2)
        out1 = out[1]  # (1,410,302,4)
        # pdb.set_trace()
        ###get the coordinates in original image

        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold[0])
        # stop = time()
        # print(str(stop-start) + "second for Generate BBX")

        # inter-scale nms
        # start = time()
        pick = nms(boxes.copy(), 0.3, 'Union')
        # stop = time()
        # print(str(stop-start) + "second for NMS inter scale")
        # start = time()
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)
        # stop = time()
    # print(str(stop-start) + "second for part 2")
    # stop1 = time()
    # print(str(stop1-start1) + "second for first stage")
    ##second stage time start
    # start2 = time()
    # img11 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rectangles = total_boxes
    # for i in range(rectangles.shape[0]):
    # cv2.rectangle(img11, (int(rectangles[i][0]),int(rectangles[i][1])),
    # (int(rectangles[i][2]),int(rectangles[i][3])), [0, 0, 255], 2)
    ##cv2.putText(img11,'%.2f'%rectangles[i][4] ,(int(rectangles[i][0]),int(rectangles[i][1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

    # plt.imshow(img11)
    # plt.show()

    ###get the coordinates in original image with bounding box regression
    numbox = total_boxes.shape[0]
    if numbox > 0:
        # start = time()
        pick = nms(total_boxes.copy(), 0.3, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        # stop = time()
        # print(str(stop-start) + "second for get original bbx with regression")

        # img11 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # rectangles = total_boxes
        # for i in range(rectangles.shape[0]):
        # cv2.rectangle(img11, (int(rectangles[i][0]),int(rectangles[i][1])),
        # (int(rectangles[i][2]),int(rectangles[i][3])), [0, 0, 255], 2)
        ##cv2.putText(img11,'%.2f'%rectangles[i][4] ,(int(rectangles[i][0]),int(rectangles[i][1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

        # plt.imshow(img11)
        # plt.show()

        total_boxes = to_rect(total_boxes.copy())  ##########box to 3:1 rectangles
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        ###############################
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        # start = time()
        tempimg = np.zeros((72, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
            :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (72, 24))
            else:
                return np.empty()
        # stop = time()
        # print(str(stop-start) + "second for resize bbx to RNet")
        # pdb.set_trace()########
        # start = time()
        tempimg = (tempimg - 127.5) * 0.0078125
        # stop = time()
        # print(str(stop-start) + "second for norlise in Rnet")

        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))

        # start = time()
        out = rnet(tempimg1)  ######################RRRRRRRRRRRRRRRRRRRRRRRR
        # stop = time()
        # print(str(stop-start) + "second for RNet running")

        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out0[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.3, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))

            img11 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rectangles = total_boxes
            # for i in range(rectangles.shape[0]):
            # cv2.rectangle(img11, (int(rectangles[i][0]),int(rectangles[i][1])),
            # (int(rectangles[i][2]),int(rectangles[i][3])), [0, 0, 255], 2)
            ## cv2.putText(img11,'%.2f'%rectangles[i][4] ,(int(rectangles[i][0]),int(rectangles[i][1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

            # plt.imshow(img11)
            # plt.show()
            ##############################################
            total_boxes = to_rect(total_boxes.copy())  ##########box to square
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            # total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
                total_boxes.copy(), w, h)
    # stop2 = time()
    # print(str(stop2-start2) + "second for second stage")
    # start3 = time()
    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        # total_boxes = np.fix(total_boxes).astype(np.int32)
        # dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
        # total_boxes.copy(), w, h)
        tempimg = np.zeros((144, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
            :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (144, 48))
            else:
                return np.empty()
        ##      pdb.set_trace()########
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = onet(tempimg1)  ############!!!!!!!!!!!!!!!!!!!!!!!!!Onet
        out0 = np.transpose(out[0])  # (2,834)
        out1 = np.transpose(out[1])  # (4,834)
        score = out0[1, :]
        ipass = np.where(score > threshold[2])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.3, 'Union')
            total_boxes = total_boxes[pick, :]
        # total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
        # total_boxes = to_rect(total_boxes.copy())
    # stop3 = time()
    # print(str(stop3-start3) + "second for third stage")

    return total_boxes


def to_rect(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    for i in range(len(h)):
        if w[i] > (h[i] / 3.0):
            bboxA[:, 1][i] = bboxA[:, 1][i] - 0.5 * (3 * w[i] - h[i])  # y1
            bboxA[:, 3][i] = bboxA[:, 3][i] + 0.5 * (3 * w[i] - h[i])  # y2
        if w[i] < (h[i] / 3.0):
            bboxA[:, 0][i] = bboxA[:, 0][i] - 0.5 * (h[i] / 3.0 - w[i])  # x1
            bboxA[:, 2][i] = bboxA[:, 2][i] + 0.5 * (h[i] / 3.0 - w[i])  # x2

    # size = np.maximum(w, h/3)
    # bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - size * 0.5
    # bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - 3*size * 0.5
    # bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(size, (2, 1)))
    return bboxA


def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    s_sort = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while s_sort.size > 0:
        i = s_sort[-1]
        pick[counter] = i
        counter += 1
        idx = s_sort[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        s_sort = s_sort[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def bbreg(boundingbox, reg):  ###bounding box regresion

    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w  # offset_x1
    b2 = boundingbox[:, 1] + reg[:, 1] * h  # offset_y1
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):
    stride = 2  #
    cellsize_x = 12  #
    cellsize_y = 36
    # cellsize = 12
    imap = np.transpose(imap)
    # offset
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    # get
    row, col = np.where(imap >= t)
    if row.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(row, col)]
    reg = np.transpose(np.vstack([dx1[(row, col)], dy1[(row, col)],
                                  dx2[(row, col)], dy2[(row, col)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    pos = np.transpose(np.vstack([row, col]))  # pos in the output feature map
    # q1 = np.fix((stride * pos + 1) / scale)
    # q2 = np.fix((stride * pos + cellsize - 1 + 1) / scale)
    bx1 = np.fix((stride * row + 1) / scale)
    by1 = np.fix((stride * col + 1) / scale)
    bbx1 = np.transpose(np.vstack([bx1, by1]))
    bx2 = np.fix((stride * row + cellsize_x) / scale)
    by2 = np.fix((stride * col + cellsize_y) / scale)
    bbx2 = np.transpose(np.vstack([bx2, by2]))

    boundingbox = np.hstack([bbx1, bbx2, np.expand_dims(score, 1), reg])
    # boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

    return boundingbox, reg


def pad(total_boxes, w, h):
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data


def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

