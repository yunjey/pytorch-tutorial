import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
    def forward(self, images):
        """Extract image feature vectors."""
        features = self.resnet(images)
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set hyper-parameters and build layers."""
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def init_weights(self):
        pass
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generate caption."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = rnn_utils.pack_padded_sequence(embeddings, lengths, batch_first=True) # lengths is ok
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, feature, state):
        """Sample a caption for given a image feature."""
        # (batch_size, seq_length, embed_size)
        # features: (1, 128)
        sampled_ids = []
        input = feature.unsqueeze(1)
        for i in range(20):
            hidden, state = self.lstm(input, state)  # (1, 1, 512)
            output = self.linear(hidden.view(-1, self.hidden_size))  # (1, 10000)
            predicted = output.max(1)[1]
            sampled_ids.append(predicted)
            input = self.embed(predicted)
        return sampled_ids