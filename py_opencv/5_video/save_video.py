# coding=utf-8
from __future__ import absolute_import, division, print_function
import cv2
import warnings
import numpy as np
warnings.simplefilter("always")


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.avi'):  # 保证文件名的后缀是.avi
            name += '.avi'
            warnings.warn('video name should ends with ".avi"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()


def main():
    width = 128
    height = 128
    vw = VideoWriter('./test.avi', width, height)
    for i in range(25*50):
        # 随机生成一幅图像
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # 写入图像
        vw.write(frame)
    # 关闭
    vw.close()


if __name__ == '__main__':
    main()
