import os
import numpy as np
import torch
import torchvision.transforms as T
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from model import EncoderCNN, DecoderRNN
from vocab import Vocabulary
from torch.autograd import Variable

# Image processing 
transform = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Hyper Parameters
embed_size = 128
hidden_size = 512
num_layers = 1

# Load vocabulary
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    
# Load an image array
images = os.listdir('./data/train2014resized/')
image_path = './data/train2014resized/' + images[12] 
img = Image.open(image_path)
image = transform(img).unsqueeze(0) 

# Load the trained models
encoder = torch.load('./encoder.pkl')
decoder = torch.load('./decoder.pkl')

# Encode the image
feature = encoder(Variable(image).cuda())

# Set initial states
state = (Variable(torch.zeros(num_layers, 1, hidden_size).cuda()),
         Variable(torch.zeros(num_layers, 1, hidden_size)).cuda())

# Decode the feature to caption
ids = decoder.sample(feature, state)

words = []
for id in ids:
    word = vocab.idx2word[id.data[0, 0]]
    words.append(word)
    if  word == '<end>':
        break    
caption = ' '.join(words)

# Display the image and generated caption
plt.imshow(img)
plt.show()
print (caption)