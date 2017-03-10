import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# Create a torch tensor with random normal.
x = torch.randn(5, 3)
print (x)

# Build a layer.
linear = nn.Linear(3, 2)
print (linear.weight)
print (linear.bias)

# Forward propagate.
y = linear(Variable(x))
print (y)

# Convert numpy array to torch tensor.
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a)
print (b)

# Download and load cifar10 dataset .
train_dataset = dsets.CIFAR10(root='./data/',
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

# Select one data pair.
image, label = train_dataset[0]
print (image.size())
print (label)

# Input pipeline (this provides queue and thread in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=2)

# When iteration starts, queue and thread start to load dataset.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of data loader is as below.
for images, labels in train_loader:
    # Your training code will be written here
    pass

# Build custom dataset.
class CustomDataset(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using np.fromfile, PIL.Image.open).
        # 2. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=2)


# Download and load pretrained model.
resnet = torchvision.models.resnet18(pretrained=True)

# Detach top layer for finetuning.
sub_model = nn.Sequential(*list(resnet.children())[:-1])

# For test
images = Variable(torch.randn(10, 3, 256, 256))
print (resnet(images).size())
print (sub_model(images).size())

# Save and load the model.
torch.save(sub_model, 'model.pkl')
model = torch.load('model.pkl')