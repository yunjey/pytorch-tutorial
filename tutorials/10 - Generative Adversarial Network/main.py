import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Image Preprocessing
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transform,
                            download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = F.sigmoid(self.fc3(h))
        return out

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 784)
            
    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        out = F.tanh(self.fc3(h))
        return out

discriminator = Discriminator()
generator = Generator()



# Loss and Optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0005)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)

# Training 
for epoch in range(200):
    for i, (images, _) in enumerate(train_loader):
        # Build mini-batch dataset
        images = images.view(images.size(0), -1)
        images = Variable(images)
        real_labels = Variable(torch.ones(images.size(0)))
        fake_labels = Variable(torch.zeros(images.size(0)))
        
        # Train the discriminator
        discriminator.zero_grad()
        outputs = discriminator(images)
        real_loss = criterion(outputs, real_labels)
        real_score = outputs
        
        noise = Variable(torch.randn(images.size(0), 128))
        fake_images = generator(noise)
        outputs = discriminator(fake_images) 
        fake_loss = criterion(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train the generator 
        generator.zero_grad()
        noise = Variable(torch.randn(images.size(0), 128))
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
                  'D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.cpu().data.mean()))
            
    # Save the sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(fake_images.data, 
        './data/fake_samples_%d.png' %(epoch+1))

# Save the Models 
torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')