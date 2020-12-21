import sys
import cv2

print(cv2.__version__)
if __name__ == '__main__':
    # 使用多线程加速
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    # 读取图片
    # im = cv2.imread(sys.argv[1])
    im = cv2.imread("/Users/binyu/Downloads/dogs-vs-cats/test1/9999.jpg")
    # resize图片
    newHeight = 200
    newWidth = int(im.shape[1] * 200 / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    # 这行代码创建一个Selective Search Segmentation对象，使用默认的参数。
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # 设置应用算法的图片
    ss.setBaseImage(im)

    ss.switchToSelectiveSearchFast()
    # fast模式，速度快，但是召回率低
    # if (sys.argv[2] == 'f'):
    #     ss.switchToSelectiveSearchFast()
    # # 高召回率但是速度慢
    # elif (sys.argv[2] == 'q'):
    #     ss.switchToSelectiveSearchQuality()
    # else:
    #     print(__doc__)
    #     sys.exit(1)

    # 实际运行算法
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # 只显示100个区域
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 1

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0),
                              1, cv2.LINE_AA)
            else:
                break

        # # show output
        # cv2.imshow("Output", imOut)

        cv2.imshow('Output', imOut)
        cv2.waitKey(0)
        cv2.destroyAllWindows()