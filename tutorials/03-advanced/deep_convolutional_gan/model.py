import torch.nn as nn 
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)      # If image_size is 64, output shape is as below.
        out = self.fc(z)                            # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(out))             # (?, 3, 64, 64)
        return out
    
    
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):                         # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = self.fc(out).squeeze()
        return out