import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import random
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from skimage.color import rgb2gray

class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        D_width = 128
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.0)
        self.ReLU = nn.ReLU()

        # Policy Memory
        self.log_probs = Variable(torch.Tensor())
        self.rewards = []
        self.values = []

        # Define Policy Gradient Network
        self.L1_p = nn.Linear(D_in, D_width)
        self.L2_p = nn.Linear(D_width, D_out)

        # Define Value Network
        self.L1_v = nn.Linear(D_in, D_width)
        self.L2_v = nn.Linear(D_width, 1)

    def forward(self, X):
        # Calculate Policy
        X_p = self.ReLU(self.drop(self.L1_p(X)))
        X_p = self.soft(self.L2_p(X_p))

        # Calculate Value
        X_v = self.ReLU(self.drop(self.L1_v(X)))
        X_v = self.L2_v(X_v)

        return X_p, X_v

# Define the agent ----------------------------------------------------------------------------------------------------

class Agent:

    def __init__(self, sensory_space, action_space, learning_rate, discount=0.99):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.steps = 0

        # Create the MDP's hyperparameters
        self.discount = discount
        self.episode_length = episode_length

        # Create neural network hyperparameters
        self.learning_rate = learning_rate

        # Create Q-learning networks and optimizer
        self.A2C_net = Net(self.sensory_space, action_space)
        self.optimizer = torch.optim.AdamW(self.A2C_net.parameters(), lr=self.learning_rate)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space)
        self.last_action = None
        self.last_value = 0.

    def reset_last_state_action(self):
        self.steps = 0

    def get_action(self, state):
        state = state.unsqueeze(0).clone()
        actions, v = self.A2C_net(Variable(state))
        m = Categorical(actions)
        action = m.sample()
        if self.A2C_net.log_probs.dim() != 0: # if the log probabilities are not empty
            self.A2C_net.log_probs = torch.cat([self.A2C_net.log_probs, m.log_prob(action)]) # append new value
        else:
            self.A2C_net.log_probs = m.log_prob(action) # make first entry

        v = v.detach()
        self.A2C_net.values.append(v)

        return action.item(), v

    def train_policy(self):
        R = 0
        returns = []

        for r in self.A2C_net.rewards[::-1]:
            R = r + self.discount * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        Advantage = returns - torch.tensor(self.A2C_net.values)
        policy_loss = torch.mean(-self.A2C_net.log_probs * Variable(Advantage), -1)
        value_loss = 0.5 * Advantage.pow(2).mean()
        ac_loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        # Reset episode history
        self.A2C_net.rewards = []
        self.A2C_net.values = []
        self.A2C_net.log_probs = Variable(torch.Tensor())

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch
        sensors = torch.from_numpy(sensors).type(torch.FloatTensor)
        last_reward = torch.tensor(last_reward).type(torch.FloatTensor)

        # Store reward from last action
        if self.steps != 0:
            self.A2C_net.rewards.append(last_reward)

        # If the episode has finished
        self.steps += 1
        if self.steps % self.episode_length == 0 and self.steps > 0:
            self.train_policy()

        # Get the policies action
        action, _ = self.get_action(sensors)

        return action

for j in range(1, 11):

    n_record_episodes = 0

    n_episodes = 2000
    episode_length = 1500
    episode_lens = []
    episode_rewards = []

    Player = Agent(4, 2, 0.001, discount=0.99)

    env = gym.make('CartPole-v1')

    T0 = time.time()
    for i_episode in range(n_episodes):
        observation = env.reset()

        total_reward = 0
        last_reward = 0.
        for t in range(episode_length):
            action = Player.learning_step(observation, last_reward)
            observation, reward, done, info = env.step(action)
            last_reward = reward
            total_reward += reward
            if done or t == episode_length - 1:
                print("Episode finished after {} timesteps".format(t + 1), i_episode, total_reward)
                episode_lens.append(t + 1)
                episode_rewards.append(total_reward)
                Player.A2C_net.rewards.append(last_reward)
                Player.train_policy()
                break
        Player.reset_last_state_action()
    print(time.time() - T0)

    env.close()

    episode_lens = np.array(episode_lens)
    episode_rewards = np.array(episode_rewards)
    res = np.zeros((episode_lens.shape[0], 2))
    res[:, 0] = episode_lens
    res[:, 1] = episode_rewards
    np.savetxt("CartPole_A2C"+str(j)+".csv", res, delimiter=",")

    # plt.plot(episode_lens)
    # plt.show()
    #
    # plt.plot(episode_rewards)
    # plt.show()