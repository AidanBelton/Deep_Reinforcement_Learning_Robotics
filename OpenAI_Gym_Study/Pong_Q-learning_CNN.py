import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import random
import numpy as np
import torch.nn as nn
import torch
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

        n_K1 = 16
        n_K2 = 32

        S1 = (8, 8)
        S2 = (4, 4)

        Stride1 = (8, 8)
        Stride2 = (2, 2)

        # Calculate output after Convolutional filters
        H1, W1 = Conv_Dims(H, W, S1, stride=Stride1)
        H2, W2 = Conv_Dims(H1, W1, S2, stride=Stride2)

        self.C1 = nn.Conv2d(in_channels, n_K1, S1, stride=Stride1)
        self.C2 = nn.Conv2d(n_K1, n_K2, S2, stride=Stride2)
        self.L1 = nn.Linear(H2*W2*n_K2, 256)
        self.L2 = nn.Linear(256, D_out)

    def forward(self, X):
        y = self.ReLU(self.C1(X))
        y = self.ReLU(self.C2(y))
        y = self.ReLU(self.L1(y.view(X.shape[0], -1)))
        y = self.L2(y)
        return y

# Define replay memory ------------------------------------------------------------------------------------------------
class Replay_Memory:

    def __init__(self, maximum_size, H, W, D, batch_size=64):
        self.maximum_size = maximum_size
        self.H = H
        self.W = W
        self.D = D
        self.write_head = 0
        self.fully_written = False
        self.batch_size = batch_size

        # Define memory blocks
        self.states = torch.zeros((self.maximum_size, self.H, self.W, self.D))
        self.actions = torch.zeros((self.maximum_size, 1))
        self.rewards = torch.zeros((self.maximum_size, 1))
        self.future_states = torch.zeros((self.maximum_size, self.H, self.W, self.D))

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

    def __init__(self, H, W, D, action_space, learning_rate, annealing_rate,
                 epsilon, discount=0.99, batch_size=64, replay_memory=10000, update_freq=10000):
        # Create robots parameters, number of sensors and number of actions
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
        self.on_policy_net = Net(H, W, D, self.action_space)
        self.off_policy_net = Net(H, W, D, self.action_space)
        self.on_policy_net.load_state_dict(self.on_policy_net.state_dict())
        self.on_policy_net.eval()
        self.optimizer = torch.optim.AdamW(self.off_policy_net.parameters(), self.learning_rate, weight_decay=0)
        self.criterion = nn.MSELoss()

        # GPU implementation
        self.on_policy_net = self.on_policy_net.to("cuda")
        self.off_policy_net = self.off_policy_net.to("cuda")

        # Create memory replay
        self.memory_replay = Replay_Memory(replay_memory, D, H, W, batch_size=self.batch_size)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros((D, H, W))
        self.last_action = None
        self.sliding_memory = torch.zeros((D, H, W))
        self.loss_record = []

    def reset_last_state_action(self):
        self.last_state = torch.zeros((D, H, W))
        self.last_action = None
        self.sliding_memory = torch.zeros((D, H, W))

    def get_action(self, state):
        if np.random.uniform() < max(-self.steps*self.annealing_rate + 1, self.epsilon):
            Y = torch.randint(0, self.action_space, (1, 1)).squeeze()
        else:
            self.on_policy_net.eval()
            with torch.no_grad():
                state = state.unsqueeze(0).to("cuda")
                Y = torch.argmax(self.on_policy_net(state)).squeeze()
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

        sensors = torch.from_numpy(sensors).type(torch.FloatTensor).unsqueeze(0)
        last_reward = torch.tensor(last_reward).type(torch.FloatTensor)

        # shift sliding memory
        temp = self.sliding_memory[0:self.sliding_memory.shape[0]-1].clone()
        self.sliding_memory[1:self.sliding_memory.shape[0]] = temp
        self.sliding_memory[0] = sensors[0].clone()

        # Store state-action-reward-state tuple in memory replay
        if self.last_action is not None:
            self.memory_replay.push(self.last_state, self.last_action, last_reward, self.sliding_memory)

        # If enough data train the off policy network
        if len(self.memory_replay) > self.batch_size:
            self.train()

        # Get the policies action
        action = self.get_action(self.sliding_memory)

        # Update last state and action for state-action-reward-state tuple
        self.last_state = self.sliding_memory.clone()
        self.last_action = action

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.on_policy_net.load_state_dict(self.off_policy_net.state_dict())
            self.on_policy_net.eval()

        return action

n_record_episodes = 30

max_val = 0.95
min_val = 0.326

n_episodes = 10000
episode_length = 10000
last_reward = 0.
episode_lens = []
episode_rewards = []

H = 84
W = 84
D = 4

Player = Agent(H, W, 4, 2, 0.001, 1/(n_episodes*750), 0.05, discount=0.99, batch_size=32,
               replay_memory=100000, update_freq=100)

env = gym.make('PongNoFrameskip-v4')
env = MaxAndSkipEnv(env)
env = FireResetEnv(env)

# Record first set of episodes
video_recorder = VideoRecorder(env, "./video/Pong_Q_CNN_untrained.mp4", enabled=True)
for i_episode in range(n_record_episodes):
    observation = env.reset()
    observation = rgb2gray(observation[30:195, 0:160])
    observation = (resize(observation, (H, W)) - min_val)/(max_val - min_val)

    total_reward = 0
    for t in range(episode_length):
        env.render()
        video_recorder.capture_frame()

        action = Player.learning_step(observation, last_reward) + 2
        observation, reward, done, info = env.step(action)
        observation = (resize(rgb2gray(observation[30:195, 0:160]), (H, W)) - min_val)/(max_val - min_val)
        last_reward = reward
        total_reward += reward
        if done or t == episode_length-1:
            print("Episode finished after {} timesteps".format(t+1), total_reward)
            episode_lens.append(t + 1)
            episode_rewards.append(total_reward)
            break
    Player.reset_last_state_action()
video_recorder.close()

T0 = time.time()
for i_episode in range(n_episodes):
    observation = env.reset()
    observation = rgb2gray(observation[30:195, 0:160])
    observation = (resize(observation, (H, W)) - min_val)/(max_val - min_val)

    total_reward = 0
    for t in range(episode_length):
        action = Player.learning_step(observation, last_reward) + 2
        observation, reward, done, info = env.step(action)
        observation = (resize(rgb2gray(observation[30:195, 0:160]), (H, W)) - min_val)/(max_val - min_val)

        last_reward = reward
        total_reward += reward
        if done or t == episode_length-1:
            avg_loss = np.mean(np.array(Player.loss_record)[len(Player.loss_record)-t:len(Player.loss_record) - 1])
            print("Episode finished after {} timesteps".format(t+1), total_reward, i_episode, avg_loss)
            episode_lens.append(t + 1)
            episode_rewards.append(total_reward)

            if i_episode % 100 == 0:
                print("saving")
                torch.save(Player.on_policy_net.state_dict(), "PongCheckpoint_"+str(i_episode))

            break
    Player.reset_last_state_action()
print(time.time() - T0)

# Record last set of episodes
video_recorder = VideoRecorder(env, "./video/Pong_Q_CNN_trained.mp4", enabled=True)
for i_episode in range(n_record_episodes):
    observation = env.reset()
    observation = rgb2gray(observation[30:195, 0:160])
    observation = (resize(observation, (H, W)) - min_val)/(max_val - min_val)

    total_reward = 0
    for t in range(episode_length):
        env.render()
        video_recorder.capture_frame()

        action = Player.learning_step(observation, last_reward) + 2
        observation, reward, done, info = env.step(action)
        observation = (resize(rgb2gray(observation[30:195, 0:160]), (H, W)) - min_val)/(max_val - min_val)
        last_reward = reward
        total_reward += reward
        if done or t == episode_length-1:
            print("Episode finished after {} timesteps".format(t+1), total_reward)
            episode_lens.append(t + 1)
            episode_rewards.append(total_reward)
            break
    Player.reset_last_state_action()
video_recorder.close()






env.close()

episode_lens = np.array(episode_lens)
episode_rewards = np.array(episode_rewards)
res = np.zeros((episode_lens.shape[0], 2))
res[:, 0] = episode_lens
res[:, 1] = episode_rewards
np.savetxt("Pong_Q_learning_CNN.csv", res, delimiter=",")

# plt.plot(episode_lens)
# plt.show()
#
# plt.plot(episode_rewards)
# plt.show()

