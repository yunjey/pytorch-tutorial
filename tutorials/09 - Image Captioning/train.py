from data import get_data_loader 
from vocab import Vocabulary
from configuration import Config
from model import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import numpy as np 
import pickle 
import os


def main():
    # Configuration for hyper-parameters
    config = Config()
    
    # Create model directory
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    
    # Image preprocessing
    transform = config.train_transform
    
    # Load vocabulary wrapper
    with open(os.path.join(config.vocab_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    image_path = os.path.join(config.image_path, 'train2014')
    json_path = os.path.join(config.caption_path, 'captions_train2014.json')
    train_loader = get_data_loader(image_path, json_path, vocab, 
                                   transform, config.batch_size,
                                   shuffle=True, num_workers=config.num_threads) 
    total_step = len(train_loader)

    # Build Models
    encoder = EncoderCNN(config.embed_size)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, 
                         len(vocab), config.num_layers)
    
    if torch.cuda.is_available()
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    
    # Train the Models
    for epoch in range(config.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % config.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, config.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the Model
            if (i+1) % config.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(config.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(config.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    main()