import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as	np
import matplotlib.pyplot as plt
import sys

model_name = sys.argv[1]
model = torch.load(model_name)
test_index = 20
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
digit, label = next(iter(test_loader))
value =  digit.numpy()
#print value[1][0]
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

plt.imshow(value[test_index][0])
plt.show()

net = Net(input_size, hidden_size, num_classes)

net.load_state_dict(model)
val = Variable(digit[test_index].view(-1,28*28))
output = net(val)
_, output = torch.max(net(val).data,1)
print "The Predicted label is "int(output.numpy())