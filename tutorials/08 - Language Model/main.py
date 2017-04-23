# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from data_utils import Dictionary, Corpus

# Hyper Parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000   # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load Penn Treebank Dataset
train_path = './data/train.txt'
sample_path = './sample.txt'
corpus = Corpus()
ids = corpus.get_data(train_path, batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

# RNN Based Language Model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x) 
        
        # Forward propagate RNN  
        out, h = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time step
        out = self.linear(out)  
        return out, h
    
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated Backpropagation 
def detach(states):
    return [state.detach() for state in states] 

# Training
for epoch in range(num_epochs):
    # Initial hidden and memory states
    states = (Variable(torch.zeros(num_layers, batch_size, hidden_size)),
              Variable(torch.zeros(num_layers, batch_size, hidden_size)))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get batch inputs and targets
        inputs = Variable(ids[:, i:i+seq_length])
        targets = Variable(ids[:, (i+1):(i+1)+seq_length].contiguous())
        
        # Forward + Backward + Optimize
        model.zero_grad()
        states = detach(states)
        outputs, states = model(inputs, states) 
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                   (epoch+1, num_epochs, step, num_batches, loss.data[0], np.exp(loss.data[0])))

# Sampling
with open(sample_path, 'w') as f:
    # Set intial hidden ane memory states
    state = (Variable(torch.zeros(num_layers, 1, hidden_size)),
         Variable(torch.zeros(num_layers, 1, hidden_size)))

    # Select one word id randomly
    prob = torch.ones(vocab_size)
    input = Variable(torch.multinomial(prob, num_samples=1).unsqueeze(1),
                     volatile=True)

    for i in range(num_samples):
        # Forward propagate rnn 
        output, state = model(input, state)
        
        # Sample a word id
        prob = output.squeeze().data.exp()
        word_id = torch.multinomial(prob, 1)[0]
        
        # Feed sampled word id to next time step
        input.data.fill_(word_id)
        
        # File write
        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i+1) % 100 == 0:
            print('Sampled [%d/%d] words and save to %s'%(i+1, num_samples, sample_path))

# Save the Trained Model
torch.save(model.state_dict(), 'model.pkl')
