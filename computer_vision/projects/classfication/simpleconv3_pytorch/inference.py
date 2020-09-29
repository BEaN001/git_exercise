import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F

from computer_vision.projects.classfication.simpleconv3_pytorch.net import simpleconv3

data_transforms =  transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

net = simpleconv3()
net.eval()
modelpath = "/Users/binyu/Documents/git_exercise/computer_vision/projects/classfication/simpleconv3_pytorch/models/model.ckpt"
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))

imagepath = "/Users/binyu/Documents/git_exercise/computer_vision/datas/mouth/1/1smile.jpg"
image = Image.open(imagepath)
imgblob = data_transforms(image).unsqueeze(0)
imgblob = Variable(imgblob)

torch.no_grad()

predict = F.softmax(net(imgblob))
print(predict)