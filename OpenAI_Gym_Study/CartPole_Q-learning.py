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

# Define the neural network module ------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        D_width = 50
        self.ReLU = nn.ReLU()

        self.L1 = nn.Linear(D_in, D_width)
        self.L2 = nn.Linear(D_width, D_out)

    def forward(self, X):
        X = self.ReLU(self.L1(X))
        X = self.L2(X)
        return X

# Define replay memory ------------------------------------------------------------------------------------------------
class Replay_Memory:

    def __init__(self, maximum_size, state_space, batch_size=64):
        self.maximum_size = maximum_size
        self.state_space = state_space
        self.write_head = 0
        self.fully_written = False
        self.batch_size = batch_size

        # Define memory blocks
        self.states = torch.zeros((self.maximum_size, state_space))
        self.actions = torch.zeros((self.maximum_size, 1))
        self.rewards = torch.zeros((self.maximum_size, 1))
        self.future_states = torch.zeros((self.maximum_size, state_space))

    def __len__(self):
        if self.fully_written:
            return self.maximum_size
        else:
            return self.write_head

    def push(self, state, action, reward, future_state):
        self.states[self.write_head] = state
        self.actions[self.write_head] = action
        self.rewards[self.write_head] = reward
        self.future_states[self.write_head] = future_state

        self.write_head += 1
        if self.write_head >= self.maximum_size:
            self.write_head = 0
            self.fully_written = True

    def get_batch(self):
        # randomly sample the replay memory
        if self.fully_written:
            idx = torch.randint(0, self.maximum_size, (self.batch_size,))
        else:
            idx = torch.randint(0, self.write_head, (self.batch_size,))

        # get the batch
        b_states = self.states[idx]
        b_actions = self.actions[idx]
        b_rewards = self.rewards[idx]
        b_future_states = self.future_states[idx]

        return b_states, b_actions, b_rewards, b_future_states

# Define the agent ----------------------------------------------------------------------------------------------------
class Agent:

    def __init__(self, sensory_space, action_space, learning_rate, annealing_rate,
                 epsilon, discount=0.99, batch_size=64, replay_memory=10000, update_freq=10000):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.steps = 0

        # Create the MDP's hyperparameters
        self.discount = discount
        self.annealing_rate = annealing_rate
        self.epsilon = epsilon

        # Create neural network hyperparameters
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.learning_rate = learning_rate

        # Create networks and optimizer
        self.on_policy_net = Net(self.sensory_space, self.action_space)
        self.off_policy_net = Net(self.sensory_space, self.action_space)
        self.on_policy_net.load_state_dict(self.on_policy_net.state_dict())
        self.on_policy_net.eval()
        self.optimizer = torch.optim.Adam(self.off_policy_net.parameters(), self.learning_rate, weight_decay=0.00001)
        self.criterion = nn.SmoothL1Loss()

        # GPU implementation
        self.on_policy_net = self.on_policy_net.to("cuda")
        self.off_policy_net = self.off_policy_net.to("cuda")

        # Create memory replay
        self.memory_replay = Replay_Memory(replay_memory, self.sensory_space, batch_size=self.batch_size)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space)
        self.last_action = None

        self.loss_record = []

    def reset_last_state_action(self):
        self.last_state = torch.zeros(self.sensory_space)
        self.last_action = None

    def get_action(self, state):
        if np.random.uniform() < max(-self.steps*self.annealing_rate + 1, self.epsilon):
            Y = torch.randint(0, self.action_space, (1, 1)).squeeze()
        else:
            self.on_policy_net.eval()
            with torch.no_grad():
                state = state.unsqueeze(0).to("cuda")
                Qs = self.on_policy_net(state)
                Y = torch.argmax(Qs).squeeze()
                Y = Y.cpu()
        return Y

    def train(self):
        b_states, b_actions, b_rewards, b_futr_states = self.memory_replay.get_batch()
        b_actions = b_actions.type(torch.LongTensor)

        # GPU implementation
        b_states = b_states.to("cuda")
        b_actions = b_actions.to("cuda")
        b_rewards = b_rewards.to("cuda")
        b_futr_states = b_futr_states.to("cuda")

        # Get expected Q value from off policy network
        Q_state_action = self.off_policy_net(b_states).gather(1, b_actions)

        # Get the best expected values for the next state
        with torch.no_grad():
            Q_next_state = torch.max(self.on_policy_net(b_futr_states), 1)[0]

        # Calculate the expecte Q value of the state-action pair
        expected_Q_val = Q_next_state.view(self.batch_size, -1)*self.discount + b_rewards

        # Backpropogate
        loss = self.criterion(Q_state_action, expected_Q_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_record.append(loss.item())

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch

        sensors = torch.from_numpy(sensors).type(torch.FloatTensor)
        last_reward = torch.tensor(last_reward).type(torch.FloatTensor)

        # Store state-action-reward-state tuple in memory replay
        if self.last_action is not None:
            self.memory_replay.push(self.last_state, self.last_action, last_reward, sensors)

        # If enough data train the off policy network
        if len(self.memory_replay) > self.batch_size:
            self.train()

        # Get the policies action
        action = self.get_action(sensors)

        # Update last state and action for state-action-reward-state tuple
        self.last_state = sensors.clone()
        self.last_action = action

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.on_policy_net.load_state_dict(self.off_policy_net.state_dict())
            self.on_policy_net.eval()

        action = action.numpy()
        return action

for j in range(7, 11):

    n_record_episodes = 0

    n_episodes = 2000
    episode_length = 1000
    episode_lens = []
    episode_rewards = []

    Player = Agent(4, 2, 0.001, 1/(n_episodes*10), 0.05, discount=0.99, batch_size=512, replay_memory=20000,
                   update_freq=500)

    env = gym.make('CartPole-v1')


    # Record first set of episodes
    if n_record_episodes > 0:
        video_recorder = VideoRecorder(env, "./video/CartPole_DQN_trained.mp4", enabled=True)
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
                if t < 499:
                    Player.learning_step(np.zeros(4), -1.)
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
                if t < 499:
                    Player.learning_step(np.zeros(4), -1.)
                break
        Player.reset_last_state_action()
    print(time.time() - T0)

    # Record last set of episodes
    if n_record_episodes > 0:
        video_recorder = VideoRecorder(env, "./video/CartPole_DQN_trained.mp4", enabled=True)
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
                if t < 499:
                    Player.learning_step(np.zeros(4), -1.)
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
    np.savetxt("CartPole_DQN"+str(j)+".csv", res, delimiter=",")

    # loss_values = np.array(Player.loss_record)
    # plt.plot(loss_values)
    # plt.show()

    # plt.plot(episode_lens)
    # plt.show()
    #
    # plt.plot(episode_rewards)
    # plt.show()