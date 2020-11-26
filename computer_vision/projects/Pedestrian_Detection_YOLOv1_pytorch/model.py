import torch
import torchvision

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)


class YoloModel(torch.nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
        self.net = torchvision.models.vgg19_bn(pretrained=True)
        self.net.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),  # 7*7*1024
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, input):
        # for vgg
        N = input.size()[0]
        output = self.net(input)
        output = torch.nn.functional.sigmoid(output)
        return output.view(N, 7, 7, 30)


if __name__ == "__main__":
    model = YoloModel().to(device)
    print(model)
    print(model(torch.randn(1,3,224,224).to(device)).size())