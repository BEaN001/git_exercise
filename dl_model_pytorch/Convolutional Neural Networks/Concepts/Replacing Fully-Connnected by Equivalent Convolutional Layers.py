import torch

inputs = torch.tensor([[[[1., 2.],
                         [3., 4.]]]])

inputs.shape

fc = torch.nn.Linear(4, 2)

weights = torch.tensor([[1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8]])
bias = torch.tensor([1.9, 2.0])
fc.weight.data = weights
fc.bias.data = bias

torch.relu(fc(inputs.view(-1, 4)))

print(torch.relu(fc(inputs.view(-1, 4))))

# We can obtain the same outputs if we use convolutional layers where the kernel size is the same size as the input feature array:
conv = torch.nn.Conv2d(in_channels=1,
                       out_channels=2,
                       kernel_size=inputs.squeeze(dim=(0)).squeeze(dim=(0)).size())
print(conv.weight.size())
print(conv.bias.size())

conv.weight.data = weights.view(2, 1, 2, 2)
conv.bias.data = bias

torch.relu(conv(inputs))
print(torch.relu(conv(inputs)))


# Similarly, we can replace the fully connected layer using a convolutional layer when we reshape the input image into a num_inputs x 1 x 1 image:
conv = torch.nn.Conv2d(in_channels=4,
                       out_channels=2,
                       kernel_size=(1, 1))

conv.weight.data = weights.view(2, 4, 1, 1)
conv.bias.data = bias
torch.relu(conv(inputs.view(1, 4, 1, 1)))
print(torch.relu(conv(inputs.view(1, 4, 1, 1))))






















