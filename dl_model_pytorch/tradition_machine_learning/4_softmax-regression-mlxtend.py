from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 0
learning_rate = 0.05
num_epochs = 10
batch_size = 8

# Architecture
num_features = 2
num_classes = 3

##########################
### DATASET
##########################

data = np.genfromtxt('../data_new/iris.data', delimiter=',', dtype=str)
X, y = data[:, [2, 3]], data[:, 4]
X = X.astype(float)

d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([d[x] for x in y])
y = y.astype(np.int)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)

# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

##########################
### Data Loaders
##########################

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        training_example, training_label = self.X[index], self.y[index]
        return training_example, training_label

    def __len__(self):
        return self.y.shape[0]


train_dataset = MyDataset(X[:100], y[:100])
test_dataset = MyDataset(X[100:], y[100:])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,  # want to shuffle the dataset
                          num_workers=4)  # number processes/CPUs to use

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4)


##########################
### MODEL
##########################

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)

model.to(device)

##########################
### COST AND OPTIMIZER
##########################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Manual seed for deterministic data loader
torch.manual_seed(random_seed)


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


for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)

        # note that the PyTorch implementation of
        # CrossEntropyLoss works with logits, not
        # probabilities
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_dataset) // batch_size, cost))

    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
            epoch + 1, num_epochs,
            compute_accuracy(model, train_loader)))

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

#Creating a ModelWrapper class for plot_decision_regions
class ModelWrapper():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, X):
        features = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits, probas = self.model(features)
        _, predicted_labels = torch.max(probas, 1)

        return predicted_labels.numpy()

mymodel = ModelWrapper(model, device=torch.device('cpu'))

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, y, mymodel)
plt.show()





