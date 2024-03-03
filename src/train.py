from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population
import datetime
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
nb_neurons = 512

DQN_model = torch.nn.Sequential(
                          nn.Linear(state_dim, nb_neurons),
                          nn.SiLU(),
                          #nn.Dropout(0.1),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.SiLU(),
                          #nn.Dropout(0.1),
                          nn.Linear(nb_neurons, nb_neurons), 
                          nn.SiLU(),
                          nn.Dropout(0.05),
                          nn.Linear(nb_neurons, n_action)
                            ).to(device)

config = {'gamma': 0.975, 
          'batch_size': 1024,
          'epsilon_max': 1.,
          'epsilon_min': 0.03,
          'epsilon_stop': 18000,
          'epsilon_delay': 10,
          'buffer_size': int(1e5),
          'learning_rate': 0.001,
          'update_target_strategy': 'ema',
          'update_target_freq': 600,
          'update_target_tau': 0.001,
          'monitor_every': 25,
          'gradient_steps': 2
}

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
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
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_stop'] if 'epsilon_stop' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay'] if 'epsilon_delay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = DQN_model 
        self.target_model = deepcopy(self.model).to(device)
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.008
        self.monitoring_nb_trials = config["monitoring_nb_trials"] if "monitoring_nb_trials" in config.keys() else 20
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 40
        self.save_every = config['save_every'] if 'save_every' in config.keys() else self.monitor_every
        

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
                return torch.argmax(Q).item()
        

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.act(x, use_random=False)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)

    def train(self, env, max_episode):
        print("Beginning training...")
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode_return = []
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        best_return = 0
        step = 0
        episode = 0

        while episode < max_episode:
            # update epsilon
            if episode > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
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
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0 and episode % self.monitor_every == 0: 
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode: ", '{:2d}'.format(episode), 
                          ", epsilon: ", '{:6.2f}'.format(epsilon), 
                          ", memory size: ", '{:4d}'.format(len(self.memory)), 
                          ", ep return / 1e5: ", '{:6.0f}'.format(episode_cum_reward/100000), 
                          ", MC tot: ", '{:6.0f}'.format(MC_tr),
                          ", MC disc: ", '{:6.0f}'.format(MC_dr),
                          ", V0: ", '{:6.0f}'.format(V0),
                          sep='')
                    if MC_tr > best_return:
                        best_return = MC_tr
                        current_time = datetime.datetime.now().strftime('%H:%M:%S')
                        print("New best return: ", best_return)
                        score_agent: float = evaluate_HIV(agent=self, nb_episode=1)
                        score_agent_dr: float = evaluate_HIV_population(agent=self, nb_episode=15) 
                        print("Score agent: {}\nScore Agent Dr: {}".format(score_agent, score_agent_dr))
                        if score_agent > 2e10: print("--------------------\n--------------------\nALLLEEELEUIA\n \
                                                     at hour:"+ current_time +"+ --------------------\n--------------------\n ")
                        if score_agent_dr > 1e10: print("--------------------\n--------------------\nALLLEEELEUIA drdrdr\n \
                                                     at hour:"+ current_time +"+ --------------------\n--------------------\n ")
                        if score_agent > 1e10: self.save("model_"+str(current_time)+".pt")
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode: ", '{:2d}'.format(episode), 
                          ", epsilon: ", '{:6.2f}'.format(epsilon), 
                          ", memory size: ", '{:4d}'.format(len(self.memory)), 
                          ", ep return / 1e5: ", '{:6.0f}'.format(episode_cum_reward/100000), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return


    def save(self, path):
        cwd_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(os.path.dirname(cwd_path), path)
        print("Saving model to: "+full_path)
        torch.save(self.model, full_path)
        print("Saved successfully")

    def load(self):
        filename = "model_final.pt"
        cwd_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(os.path.dirname(cwd_path), filename)
        print("Trying to load model file"+full_path)
        self.model = torch.load(full_path, map_location=torch.device("cpu"))
        self.memory.to(torch.device('cpu'))
        self.device = torch.device('cpu')
    