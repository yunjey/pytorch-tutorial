import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Image Preprocessing
transform = transforms.Compose([
        transforms.Scale(36),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='../data/',
                               train=True, 
                               transform=transform,
                               download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

# 4x4 Convolution
def conv4x4(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                     stride=stride, padding=1, bias=False)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            conv4x4(3, 16, 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv4x4(16, 32, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            conv4x4(32, 64, 2), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4),
            nn.Sigmoid())
    
    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        return out

# 4x4 Transpose convolution
def conv_transpose4x4(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, 
                              stride=stride, padding=padding, bias=bias)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            conv_transpose4x4(128, 64, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_transpose4x4(64, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_transpose4x4(32, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv_transpose4x4(16, 3, 2, bias=True),
            nn.Tanh())
    
    def forward(self, x):
        x = x.view(x.size(0), 128, 1, 1)
        out = self.model(x)
        return out

discriminator = Discriminator()
generator = Generator()
discriminator.cuda()
generator.cuda()

# Loss and Optimizer
criterion = nn.BCELoss()
lr = 0.002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

# Training 
for epoch in range(50):
    for i, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0))).cuda()
        fake_labels = Variable(torch.zeros(images.size(0))).cuda()
        
        # Train the discriminator
        discriminator.zero_grad()
        outputs = discriminator(images)
        real_loss = criterion(outputs, real_labels)
        real_score = outputs
        
        noise = Variable(torch.randn(images.size(0), 128)).cuda()
        fake_images = generator(noise)
        outputs = discriminator(fake_images) 
        fake_loss = criterion(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train the generator 
        generator.zero_grad()
        noise = Variable(torch.randn(images.size(0), 128)).cuda()
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
                  'D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch, 50, i+1, 500, d_loss.data[0], g_loss.data[0],
                    real_score.cpu().data.mean(), fake_score.cpu().data.mean()))
            
            # Save the sampled images
            torchvision.utils.save_image(fake_images.data, 
                './data/fake_samples_%d_%d.png' %(epoch+1, i+1))

# Save the Models 
torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')