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

# Define the neural network module ------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        D_width = 128
        self.ReLU = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.)

        self.L1 = nn.Linear(D_in, D_width)
        self.B1 = nn.LayerNorm(D_width)
        self.L2 = nn.Linear(D_width, D_out)

        # Memory
        self.log_probs = Variable(torch.Tensor())
        self.rewards = []

    def forward(self, X):
        X = self.ReLU(self.B1(self.drop(self.L1(X))))
        X = self.soft(self.L2(X))
        return X

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

        # Create networks and optimizer
        self.policy_net = Net(sensory_space, self.action_space)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), self.learning_rate, weight_decay=0.)
        self.eps = np.finfo(np.float32).eps.item()

        # GPU implementation
        self.policy_net = self.policy_net.to("cuda")

    def reset_last_state_action(self):
        self.steps = 0

    def get_action(self, state):
        state = state.unsqueeze(0).to("cuda")
        action_prob = self.policy_net(Variable(state))
        m = Categorical(action_prob)
        action = m.sample()
        if len(self.policy_net.log_probs) != 0: # if the log probabilities are not empty
            self.policy_net.log_probs = torch.cat((self.policy_net.log_probs, m.log_prob(action)), 0) # append new value
        else:
            self.policy_net.log_probs = m.log_prob(action).view(1) # make first entry

        return action.item()

    def train(self):
        R = 0
        returns = []

        for r in self.policy_net.rewards[::-1]:
            R = r + self.discount * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to("cuda")
        policy_loss = torch.mean(-self.policy_net.log_probs * Variable(returns), -1)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Reset episode history
        self.policy_net.rewards = []
        self.policy_net.log_probs = Variable(torch.Tensor())

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch
        sensors = torch.from_numpy(sensors).type(torch.FloatTensor)
        last_reward = torch.tensor(last_reward).type(torch.FloatTensor)

        # Store reward from last action
        if self.steps != 0:
            self.policy_net.rewards.append(last_reward)

        # If the episode has finished
        self.steps += 1

        # Get the policies action
        action = self.get_action(sensors)
        return action

for j in range(10, 11):

    n_record_episodes = 0

    n_episodes = 2000
    episode_length = 1500
    episode_lens = []
    episode_rewards = []

    Player = Agent(4, 2, 0.001, discount=0.99)

    env = gym.make('CartPole-v1')


    # Record first set of episodes
    if n_record_episodes > 0:
        video_recorder = VideoRecorder(env, "./video/CartPole_DPG_untrained.mp4", enabled=True)
    for i_episode in range(n_record_episodes):
        observation = env.reset()

        total_reward = 0
        last_reward = 0.
        for t in range(episode_length):
            env.render()
            video_recorder.capture_frame()

            action = Player.learning_step(observation, last_reward)
            observation, reward, done, info = env.step(action)
            last_reward = reward
            total_reward += reward
            if done or t == episode_length-1:
                print("Episode finished after {} timesteps".format(t+1), i_episode, total_reward)
                episode_lens.append(t + 1)
                episode_rewards.append(total_reward)
                Player.policy_net.rewards.append(last_reward)
                Player.train()
                break
        Player.reset_last_state_action()
    if n_record_episodes > 0:
        video_recorder.close()

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
                Player.policy_net.rewards.append(last_reward)
                Player.train()
                break
        Player.reset_last_state_action()
    print(time.time() - T0)

    # Record last set of episodes
    if n_record_episodes > 0:
        video_recorder = VideoRecorder(env, "./video/CartPole_DPG_trained.mp4", enabled=True)
    for i_episode in range(n_record_episodes):
        observation = env.reset()

        total_reward = 0
        last_reward = 0.
        for t in range(episode_length):
            env.render()
            video_recorder.capture_frame()

            action = Player.learning_step(observation, last_reward)
            observation, reward, done, info = env.step(action)
            last_reward = reward
            total_reward += reward
            if done or t == episode_length - 1:
                print("Episode finished after {} timesteps".format(t + 1), i_episode, total_reward)
                episode_lens.append(t + 1)
                episode_rewards.append(total_reward)
                Player.policy_net.rewards.append(last_reward)
                Player.train()
                break
        Player.reset_last_state_action()
    if n_record_episodes > 0:
        video_recorder.close()






    env.close()

    episode_lens = np.array(episode_lens)
    episode_rewards = np.array(episode_rewards)
    res = np.zeros((episode_lens.shape[0], 2))
    res[:, 0] = episode_lens
    res[:, 1] = episode_rewards
    np.savetxt("CartPole_Policy_Grad"+str(j)+".csv", res, delimiter=",")

    # plt.plot(episode_lens)
    # plt.show()
    #
    # plt.plot(episode_rewards)
    # plt.show()