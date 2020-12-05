import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


class DogCat(Dataset):

    def __init__(self, data_dir, transform=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        :param data_dir:
        :param train:
        :param test:
        """
        super(DogCat, self).__init__()

        self.data_dir = data_dir

        self.images_dir = []
        self.gen_imgs_dir()

        self.test = test

        if self.test:
            imgs = sorted(self.images_dir, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(self.images_dir, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transform is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                    ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                    ])
        else:
            self.trainforms = transforms.ToTensor()

    def gen_imgs_dir(self):
        """
        get absolute directory of all images
        :return:
        """
        imgs = os.listdir(self.data_dir)
        for im in imgs:
            if im == '.DS_Store':
                continue
            self.images_dir.append(os.path.join(self.data_dir, im))  # absolute dir

        print('total number of image is ' + str(len(self.images_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """

        :param idx:
        :return:
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label, img_path


# test case
if __name__ == "__main__":
    train_data_dir = "/Users/binyu/Downloads/dogs-vs-cats/train"
    train_data = DogCat(data_dir=train_data_dir)
    img, label, name = train_data[2]
    print('dataset info ...')
    print(img.shape)
    print(label)
    print(name)

    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

    test_data_dir = "/Users/binyu/Downloads/dogs-vs-cats/test1"
    test_data = DogCat(data_dir=test_data_dir, test=True)
    img, label, name = test_data[2]
    print('dataset info ...')
    print(img.shape)
    print(label)
    print(name)

    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

