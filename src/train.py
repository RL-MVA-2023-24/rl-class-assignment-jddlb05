from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.@
# ENJOY!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n
nb_neurons = 32

DQN_model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)

config = {'gamma': 0.9, 
          'batch_size': 32,
          'nb_actions': 32,
          'batch_size': 32,
          'epsilon_max': 1.,
          'epsilon_min': 0.1,
          'epsilon_decay_period': 40,
          'epsilon_decay_delay': 10,
          'buffer_size': int(1e5),
          'learning_rate':0.001
}

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self, name="agent_test1") -> None:
        print("Initializing agent...")
        self.name = name
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_decay_delay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = DQN_model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        

    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, torch.tensor([1.], device=next(self.model.parameters()).device), QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, nb_episodes, max_steps):
        print("Beginning training...")
        episode_return = []
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        for episode in range(nb_episodes):
            if episode > 0:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            while step < max_steps:
                # update epsilon

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                # step
                next_state, reward, _, _, _ = env.step(action)
                self.memory.append(state, action, reward, next_state)
                episode_cum_reward += reward

                # train
                self.gradient_step()

                # next transition
                step += 1

            print("Episode ", '{:3d}'.format(episode), 
                ", epsilon ", '{:6.2f}'.format(epsilon), 
                ", Memory length ", '{:5d}'.format(len(self.memory)), 
                ", episode return / 1e5 ", '{:4.1f}'.format(episode_cum_reward/10000),
                sep='')
            state, _ = env.reset()
            episode_return.append(episode_cum_reward)
            episode_cum_reward = 0

            step = 0

        return episode_return



    def save(self, path):
        with open(self.name+'.pkl','wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        filename = "/test_40eps.pkl"
        print("Trying to load file"+os.getcwd()+filename)
        with open(filename,'rb') as f:
            loaded_data = pickle.load(f)
            print("Loaded successfully")
        self.__dict__.update(loaded_data)
        self.model.to('cpu')
