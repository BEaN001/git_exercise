import cv2
import torch
import math
import torch.nn.functional as F
import skimage.transform as trans
import numpy as np

def blog_test():
    img = cv2.imread('/Users/binyu/Documents/git_exercise/computer_vision/datas/mouth/0/1neutral.jpg')
    out_h = img.shape[0]
    out_w = img.shape[1]
    img = np.moveaxis(img, -1, 0)
    print(img.shape)
    img_batch = torch.from_numpy(img).unsqueeze(0).float()
    # -30 表示正向采样中的30(但为顺时针，实际旋转矩阵的角度定义的是逆时针转的度数)
    angle = -30 * math.pi / 180  # 顺时针旋转30度，一定要加pi！
    #D = np.diag([1.5, 1.5])
    A = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    print('A', A)
    tx = 0
    ty = 0
    theta = np.array(
        [[A[0, 0], A[0, 1], tx], [A[1, 0], A[1, 1], ty]])
    theta = torch.from_numpy(theta).float().unsqueeze(0)

    batch_size = theta.size()[0]
    out_size = torch.Size(
        (batch_size, 3, out_h, out_w))
    # 结论！！
    # 需要注意，这个theta与一般见到的theta不一样，这个是反着来的
    grid = F.affine_grid(theta, out_size)
    warped_image_batch = F.grid_sample(img_batch, grid)
    print(warped_image_batch.shape)
    output = warped_image_batch[0, :, :,
                                :].cpu().detach().numpy().astype('uint8')
    print(output.shape)
    output = np.moveaxis(output, 0, -1)
    cv2.imshow('out', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    blog_test()