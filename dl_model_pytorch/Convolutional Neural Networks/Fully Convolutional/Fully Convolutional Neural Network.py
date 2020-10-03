import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 15
batch_size = 256

# Architecture
num_classes = 10


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='../data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../data',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


##########################
### MODEL
##########################


class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        # 28x28x1 => 28x28x4
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=4,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(28-1) - 28 + 3) / 2 = 1
        # 28x28x4 => 14x14x4
        self.conv_2 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=4,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        # 14x14x4 => 14x14x8
        self.conv_3 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(14-1) - 14 + 3) / 2 = 1
        # 14x14x8 => 7x7x8
        self.conv_4 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        # 7x7x8 => 7x7x16
        self.conv_5 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(7-1) - 7 + 3) / 2 = 1
        # 7x7x16 => 4x4x16
        self.conv_6 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)

        # 4x4x16 => 4x4xnum_classes
        self.conv_7 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=self.num_classes,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)  # (1(7-1) - 7 + 3) / 2 = 1

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)

        out = self.conv_2(out)
        out = F.relu(out)

        out = self.conv_3(out)
        out = F.relu(out)

        out = self.conv_4(out)
        out = F.relu(out)

        out = self.conv_5(out)
        out = F.relu(out)

        out = self.conv_6(out)
        out = F.relu(out)

        out = self.conv_7(out)
        out = F.relu(out)

        logits = F.adaptive_avg_pool2d(out, 1)
        # drop width
        logits.squeeze_(-1)
        # drop height
        logits.squeeze_(-1)
        probas = torch.softmax(logits, dim=1)
        return logits, probas


torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), cost))

    model = model.eval()
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
        epoch + 1, num_epochs,
        compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))



























































































































































