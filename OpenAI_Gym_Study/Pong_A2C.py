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
from baselines_wrappers import FireResetEnv, MaxAndSkipEnv

# Define the neural network module ------------------------------------------------------------------------------------
def Conv_Dims(H_in, W_in, kernel_size, stride=(1, 1), padding=(0, 0), dialation=(1, 1)):
    H_out = (H_in + 2*padding[0] - dialation[0]*(kernel_size[0] - 1) - 1)/stride[0] + 1
    W_out = (W_in + 2*padding[1] - dialation[1]*(kernel_size[1] - 1) - 1)/stride[1] + 1
    return int(H_out), int(W_out)

class Net(nn.Module):

    def __init__(self, H, W, in_channels, D_out):
        super(Net, self).__init__()
        self.D_out = D_out
        self.ReLU = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        # Policy Memory
        self.log_probs = Variable(torch.Tensor())
        self.rewards = []
        self.values = []

        n_K1 = 16
        n_K2 = 32

        S1 = (8, 8)
        S2 = (4, 4)

        Stride1 = (8, 8)
        Stride2 = (2, 2)

        # Calculate output after Convolutional filters
        H1, W1 = Conv_Dims(H, W, S1, stride=Stride1)
        H2, W2 = Conv_Dims(H1, W1, S2, stride=Stride2)

        self.C1_p = nn.Conv2d(in_channels, n_K1, S1, stride=Stride1)
        self.C2_p = nn.Conv2d(n_K1, n_K2, S2, stride=Stride2)
        self.L1_p = nn.Linear(H2*W2*n_K2, 256)
        self.L2_p = nn.Linear(256, D_out)

        # Define Value Network
        self.C1_v = nn.Conv2d(in_channels, n_K1, S1, stride=Stride1)
        self.C2_v = nn.Conv2d(n_K1, n_K2, S2, stride=Stride2)
        self.L1_v = nn.Linear(H2*W2*n_K2, 256)
        self.L2_v = nn.Linear(256, 1)

    def forward(self, X):
        # Calculate Policy
        X_p = self.ReLU(self.C1_p(X))
        X_p = self.ReLU(self.C2_p(X_p))
        X_p = self.ReLU(self.L1_p(X_p.view(X.shape[0], -1)))
        X_p = self.soft(self.L2_p(X_p))

        # Calculate Value
        X_v = self.ReLU(self.C1_v(X))
        X_v = self.ReLU(self.C2_v(X_v))
        X_v = self.ReLU(self.L1_v(X_v.view(X.shape[0], -1)))
        X_v = self.L2_v(X_v)

        return X_p, X_v

# Define the agent ----------------------------------------------------------------------------------------------------
class Agent:

    def __init__(self, H, W, D, action_space, learning_rate, discount=0.99):
        # Create robots parameters, number of sensors and number of actions
        self.H = H
        self.W = W
        self.D = D
        self.action_space = action_space
        self.steps = 0

        # Create the MDP's hyperparameters
        self.discount = discount
        self.episode_length = episode_length

        # Create neural network hyperparameters
        self.learning_rate = learning_rate

        # Create networks and optimizer
        self.A2C_net = Net(H, W, D, self.action_space)
        self.optimizer = torch.optim.AdamW(self.A2C_net.parameters(), self.learning_rate, weight_decay=0.)
        self.eps = np.finfo(np.float32).eps.item()

        self.sliding_memory = torch.zeros((D, H, W))

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

        # shift sliding memory
        temp = self.sliding_memory[0:self.sliding_memory.shape[0]-1].clone()
        self.sliding_memory[1:self.sliding_memory.shape[0]] = temp
        self.sliding_memory[0] = sensors[0].clone()

        # Store reward from last action
        if self.steps != 0:
            self.A2C_net.rewards.append(last_reward)

        # If the episode has finished
        self.steps += 1

        # Get the policies action
        action, _ = self.get_action(self.sliding_memory)
        return action

n_record_episodes = 0

n_episodes = 5000
episode_length = 10000
episode_lens = []
episode_rewards = []

H = 84
W = 84

Player = Agent(H, W, 4, 2, 0.001, discount=0.99)

env = gym.make('PongNoFrameskip-v4')
env = MaxAndSkipEnv(env)
env = FireResetEnv(env)

T0 = time.time()
for i_episode in range(n_episodes):
    observation = env.reset()
    observation = rgb2gray(observation[25:210, 0:160])
    observation = resize(observation, (H, W))

    total_reward = 0
    last_reward = 0.
    for t in range(episode_length):
        action = Player.learning_step(observation, last_reward)+2
        observation, reward, done, info = env.step(action)
        observation = resize(rgb2gray(observation[25:210, 0:160]), (H, W))
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
np.savetxt("Pong_A2C.csv", res, delimiter=",")

# plt.plot(episode_lens)
# plt.show()
#
# plt.plot(episode_rewards)
# plt.show()