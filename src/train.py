from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
from copy import deepcopy
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
                          nn.Dropout(0.1),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(nb_neurons, nb_neurons), 
                          nn.ReLU(),
                          nn.Dropout(0.1),
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
          'learning_rate':0.001,
          'update_target_strategy':'ema',
          'update_target_freq':20,
          'update_target_tau':0.008
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
    def to(self, device):
        self.device = device

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
        self.target_model = deepcopy(self.model).to(device)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.008
        

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
                return torch.argmax(Q).item()
        

        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, torch.tensor([1.], device=self.device), QYmax, value=self.gamma)
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
            if episode > 0: #decay epsilon every episode for now
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            while step < max_steps:
                # update epsilon

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state, use_random=False)

                # step
                next_state, reward, _, _, _ = env.step(action)
                self.memory.append(state, action, reward, next_state)
                episode_cum_reward += reward

                # train
                for _ in range(self.nb_gradient_steps): 
                    self.gradient_step()
                # update target network if needed
                if self.update_target_strategy == 'replace':
                    if step % self.update_target_freq == 0: 
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.update_target_strategy == 'ema':
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    tau = self.update_target_tau
                    for key in model_state_dict:
                        target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                    self.target_model.load_state_dict(target_state_dict)
                # next transition

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
        cwd_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(os.path.dirname(cwd_path), path)
        print("Saving model to: "+full_path)
        torch.save(self.model, full_path)
        print("Saved successfully")

    def load(self):
        filename = "test_200eps_1.pt"
        cwd_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(os.path.dirname(cwd_path), filename)
        print("Trying to load model file"+full_path)
        self.model = torch.load(full_path, map_location=torch.device("cpu"))
        self.memory.to(torch.device('cpu'))
    