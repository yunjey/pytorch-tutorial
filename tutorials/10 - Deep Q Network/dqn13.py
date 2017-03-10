%matplotlib inline

import torch
import torch.nn as nn
import gym
import random
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import deque, namedtuple

env = gym.envs.make("CartPole-v0")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)
        self.init_weights()
    
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out
        
def make_epsilon_greedy_policy(network, epsilon, nA):
    def policy(state):
        sample = random.random()
        if sample < (1-epsilon) + (epsilon/nA):
            q_values = network(state.view(1, -1))
            action = q_values.data.max(1)[1][0, 0]
        else:
            action = random.randrange(nA)
        return action
    return policy

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque()
        self.capacity = capacity
        
    def push(self, transition):
        if len(self.memory) > self.capacity:
            self.memory.popleft()
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
     
    def __len__(self):
        return len(self.memory)
        
def to_tensor(ndarray, volatile=False):
    return Variable(torch.from_numpy(ndarray), volatile=volatile).float()
    
def deep_q_learning(num_episodes=10, batch_size=100, 
                    discount_factor=0.95, epsilon=0.1, epsilon_decay=0.95):

    # Q-Network and memory 
    net = Net()
    memory = ReplayMemory(10000)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    for i_episode in range(num_episodes):
        
        # Set policy (TODO: decaying epsilon)
        #if (i_episode+1) % 100 == 0:
        #    epsilon *= 0.9
            
        policy = make_epsilon_greedy_policy(
            net, epsilon, env.action_space.n)
        
        # Start an episode
        state = env.reset()
        
        for t in range(10000):
            
            # Sample action from epsilon greed policy
            action = policy(to_tensor(state)) 
            next_state, reward, done, _ = env.step(action)
            
            
            # Restore transition in memory
            memory.push([state, action, reward, next_state])
            
            
            if len(memory) >= batch_size:
                # Sample mini-batch transitions from memory
                batch = memory.sample(batch_size)
                state_batch = np.vstack([trans[0] for trans in batch])
                action_batch =np.vstack([trans[1] for trans in batch]) 
                reward_batch = np.vstack([trans[2] for trans in batch])
                next_state_batch = np.vstack([trans[3] for trans in batch])
                
                # Forward + Backward + Opimize
                net.zero_grad()
                q_values = net(to_tensor(state_batch))
                next_q_values = net(to_tensor(next_state_batch, volatile=True))
                next_q_values.volatile = False
                
                td_target = to_tensor(reward_batch) + discount_factor * (next_q_values).max(1)[0]
                loss = criterion(q_values.gather(1, 
                            to_tensor(action_batch).long().view(-1, 1)), td_target)
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
            state = next_state
            
        if len(memory) >= batch_size and (i_episode+1) % 10 == 0:
            print ('episode: %d, time: %d, loss: %.4f' %(i_episode, t, loss.data[0]))