import torch.nn as nn
import torch
from computer_vision.projects.pytorchbook_best_practice.model.BasicModule import BasicModule
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    实现子module: Residual Blocks
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=outchannel),  # 批归一化
            nn.ReLU(inplace=True),  # 将计算得到的值直接覆盖之前的值,能够节省运算内存，不用多存储其他变量
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(BasicModule):
    """
    实现主module：ResNet34
    https://www.codenong.com/cs105438559/
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        # ResNet前两层与GoogleNet一样，每个卷积层多加一个BN层
        # 输出通道64，步幅为2的7*7卷积层，连接步幅为2的3*3最大池化层
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    # 构建layer,包含多个residual block
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 刚开始两个channel可能不同，所以right通过shortcut把通道也变为outchannel
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        # 之后的channel相同并且 w h也同，而经过ResidualBlock其w h不变，
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    net = ResNet34(num_classes=2)

    from torchsummary import summary
    summary(net, input_size=(3, 368, 368), batch_size=-1)

    x = torch.randn(2, 3, 368, 368)
    y = net(x)
    print(y.shape)

    # import torchvision
    # model = torchvision.models.resnet34(pretrained=False)  # 我们不下载预训练权重
    # print(model)
