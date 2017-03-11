from data import get_loader 
from vocab import Vocabulary
from models import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn 
import numpy as np 
import torchvision.transforms as T 
import pickle 

# Hyper Parameters
num_epochs = 5
batch_size = 100
embed_size = 128
hidden_size = 512
num_layers = 1
learning_rate = 0.001
train_image_path = './data/train2014resized/'
train_json_path = './data/annotations/captions_train2014.json'

# Image Preprocessing
transform = T.Compose([
    T.RandomHorizontalFlip(), 
    T.ToTensor(), 
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Load Vocabulary Wrapper
with open('./data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
# Build Dataset Loader
train_loader = get_loader(train_image_path, train_json_path, vocab, transform, 
                          batch_size=batch_size, shuffle=True, num_workers=2) 
total_step = len(train_loader)

# Build Models
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder.cuda()
decoder.cuda()
        
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

# Train the Decoder
for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = Variable(images).cuda()
        captions = Variable(captions).cuda()
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        # Forward, Backward and Optimize
        decoder.zero_grad()
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                  %(epoch, num_epochs, i, total_step, loss.data[0], np.exp(loss.data[0])))    
            
# Save the Model
torch.save(decoder, 'decoder.pkl')
torch.save(encoder, 'encoder.pkl')