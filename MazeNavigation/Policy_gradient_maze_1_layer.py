import random
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mazes import Mazes
import time

# Define the neural network module ------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.soft = nn.Softmax(dim=1)
        D_width = 100
        self.ReLU = nn.ReLU()

        self.L1 = nn.Linear(D_in, D_out)

        # Memory
        self.log_probs = Variable(torch.Tensor())
        self.rewards = []

    def forward(self, X):
        X = self.soft(self.L1(X))
        return X

# Define the agent ----------------------------------------------------------------------------------------------------
class Agent:

    def __init__(self, sensory_space, action_space, learning_rate, discount=0.99, recent_memory=64, episode_length=1000):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.recent_memory = recent_memory
        self.steps = 0

        # Create the MDP's hyperparameters
        self.discount = discount
        self.episode_length = episode_length

        # Create neural network hyperparameters
        self.learning_rate = learning_rate

        # Create networks and optimizer
        self.policy_net = Net(self.sensory_space*self.recent_memory, self.action_space)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.sensory_space*self.recent_memory)

    def get_action(self, state):
        state = state.unsqueeze(0).clone()
        action_prob = self.policy_net(Variable(state))
        m = Categorical(action_prob)
        action = m.sample()
        if self.policy_net.log_probs.dim() != 0: # if the log probabilities are not empty
            self.policy_net.log_probs = torch.cat([self.policy_net.log_probs, m.log_prob(action)]) # append new value
        else:
            self.policy_net.log_probs = m.log_prob(action) # make first entry
        return action.item()

    def train(self):
        R = 0
        returns = []

        for r in self.policy_net.rewards[::-1]:
            R = r + self.discount * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + self.eps)
        policy_loss = torch.sum(-self.policy_net.log_probs * Variable(returns), -1)

        self.optimizer.zero_grad()
        # print(policy_loss)
        policy_loss.backward()
        self.optimizer.step()

        # Reset episode history
        self.policy_net.rewards = []
        self.policy_net.log_probs = Variable(torch.Tensor())

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch
        sensors = torch.from_numpy(sensors).type(torch.FloatTensor)
        last_reward = torch.from_numpy(last_reward).type(torch.FloatTensor)

        # update sliding memory
        self.sliding_memory[0:self.sensory_space*(self.recent_memory-1)] = \
            self.sliding_memory.clone()[self.sensory_space:self.sensory_space*self.recent_memory]
        self.sliding_memory[self.sensory_space * (self.recent_memory - 1):self.sensory_space * self.recent_memory] = \
            sensors

        # Store reward from last action
        if self.steps != 0:
            self.policy_net.rewards.append(last_reward)

        # If the episode has finished
        self.steps += 1
        if self.steps % self.episode_length == 0 and self.steps > 0:
            self.train()

        # Get the policies action
        action = self.get_action(self.sliding_memory)
        return action

# Define the environment ----------------------------------------------------------------------------------------------

# list of possible actions
# [up, down, right, left]
actions = [0, 1, 2, 3]

# Choose the maze
myMazes = Mazes()
M, S, G = myMazes.get_maze(5)
initial_pos = S[0]
goal_pos = G[0]
print(initial_pos, goal_pos)

# Environment settings
max_run_time = 100000

# Initial Agent Conditions
agent_pos = initial_pos.copy()
old_agent_pos = [0, 0]
old_old_agent_pos = [0, 0]
maze_agent = Agent(8, 4, 0.01, discount=0.99, recent_memory=64, episode_length=1000)

# Final Values
FixedStartFixedEnd = 0
FixedStartMovingEnd = 0
MovingStartMovingEnd = 0

# Data collection
steps_to_goal = 0
n_goals = 0
goal_data = []
positions = []
run_into_wall = []
wasted_step = []
self_action_count = 0
steps_to_goal_ittr = []
reward = []

Reward = 0
# Get clock time
T0 = time.time()
for ittr in range(0, max_run_time):
    # positions.append(agent_pos.copy())

    # Get the agents sensory information
    senses = []
    senses.append(M[agent_pos[0]-1, agent_pos[1]])
    senses.append(M[agent_pos[0]+1, agent_pos[1]])
    senses.append(M[agent_pos[0], agent_pos[1]-1])
    senses.append(M[agent_pos[0], agent_pos[1]+1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]-1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]-1])
    senses = np.array(senses)

    # Get the agents actions
    agent_action = maze_agent.learning_step(senses, np.array(Reward))

    # Store old agents position
    old_old_agent_pos = old_agent_pos.copy()
    old_agent_pos = agent_pos.copy()

    # update position if there is no wall
    if agent_action == 0 and M[agent_pos[0] - 1, agent_pos[1]] == 1:  # move up
        agent_pos[0] -= 1
    elif agent_action == 1 and M[agent_pos[0] + 1, agent_pos[1]] == 1:  # move down
        agent_pos[0] += 1
    elif agent_action == 2 and M[agent_pos[0], agent_pos[1] + 1] == 1:  # move right
        agent_pos[1] += 1
    elif agent_action == 3 and M[agent_pos[0], agent_pos[1] - 1] == 1:  # move left
        agent_pos[1] -= 1

    if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
        Reward = 1
        goal_data.append(steps_to_goal)
        n_goals += 1
        # print("GOAL!!", steps_to_goal, n_goals, goal_pos)
        steps_to_goal_ittr.append([steps_to_goal, ittr])
        steps_to_goal = 0
        agent_pos = initial_pos.copy()
    else:
        Reward = 0
    reward.append(Reward)
    steps_to_goal += 1

print("Fixed Start Fixed End", n_goals)
FixedStartFixedEnd = n_goals

# Data collection
n_goals = 0
for ittr in range(0, max_run_time):
    # positions.append(agent_pos.copy())

    # Get the agents sensory information
    senses = []
    senses.append(M[agent_pos[0]-1, agent_pos[1]])
    senses.append(M[agent_pos[0]+1, agent_pos[1]])
    senses.append(M[agent_pos[0], agent_pos[1]-1])
    senses.append(M[agent_pos[0], agent_pos[1]+1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]-1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]-1])
    senses = np.array(senses)

    # Get the agents actions
    agent_action = maze_agent.learning_step(senses, np.array(Reward))

    # Store old agents position
    old_old_agent_pos = old_agent_pos.copy()
    old_agent_pos = agent_pos.copy()

    # update position if there is no wall
    if agent_action == 0 and M[agent_pos[0] - 1, agent_pos[1]] == 1:  # move up
        agent_pos[0] -= 1
    elif agent_action == 1 and M[agent_pos[0] + 1, agent_pos[1]] == 1:  # move down
        agent_pos[0] += 1
    elif agent_action == 2 and M[agent_pos[0], agent_pos[1] + 1] == 1:  # move right
        agent_pos[1] += 1
    elif agent_action == 3 and M[agent_pos[0], agent_pos[1] - 1] == 1:  # move left
        agent_pos[1] -= 1

    if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
        Reward = 1
        goal_data.append(steps_to_goal)
        n_goals += 1
        # print("GOAL!!", steps_to_goal, n_goals, goal_pos)
        steps_to_goal_ittr.append([steps_to_goal, ittr+max_run_time])
        steps_to_goal = 0
        agent_pos = initial_pos.copy()
        goal_pos = G[random.randint(0, 2)]
    else:
        Reward = 0
    reward.append(Reward)
    steps_to_goal += 1

print("Fixed Start Moving End", n_goals)
FixedStartMovingEnd = n_goals
goal_pos = G[0]
# Data collection
n_goals = 0
for ittr in range(0, max_run_time):
    # positions.append(agent_pos.copy())

    # Get the agents sensory information
    senses = []
    senses.append(M[agent_pos[0]-1, agent_pos[1]])
    senses.append(M[agent_pos[0]+1, agent_pos[1]])
    senses.append(M[agent_pos[0], agent_pos[1]-1])
    senses.append(M[agent_pos[0], agent_pos[1]+1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]+1])
    senses.append(M[agent_pos[0]-1, agent_pos[1]-1])
    senses.append(M[agent_pos[0]+1, agent_pos[1]-1])
    senses = np.array(senses)

    # Get the agents actions
    agent_action = maze_agent.learning_step(senses, np.array(Reward))

    # Store old agents position
    old_old_agent_pos = old_agent_pos.copy()
    old_agent_pos = agent_pos.copy()

    # update position if there is no wall
    if agent_action == 0 and M[agent_pos[0] - 1, agent_pos[1]] == 1:  # move up
        agent_pos[0] -= 1
    elif agent_action == 1 and M[agent_pos[0] + 1, agent_pos[1]] == 1:  # move down
        agent_pos[0] += 1
    elif agent_action == 2 and M[agent_pos[0], agent_pos[1] + 1] == 1:  # move right
        agent_pos[1] += 1
    elif agent_action == 3 and M[agent_pos[0], agent_pos[1] - 1] == 1:  # move left
        agent_pos[1] -= 1

    if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
        Reward = 1
        goal_data.append(steps_to_goal)
        n_goals += 1
        # print("GOAL!!", steps_to_goal, n_goals, goal_pos)
        steps_to_goal_ittr.append([steps_to_goal, ittr+max_run_time*2])
        steps_to_goal = 0
        initial_pos = S[random.randint(0, 2)]
        agent_pos = initial_pos.copy()
        # print("New position", agent_pos)
    else:
        Reward = 0
    reward.append(Reward)
    steps_to_goal += 1

# print("Moving Start Fixed End", n_goals)
MovingStartMovingEnd = n_goals

print("\n FINAL Results", FixedStartFixedEnd, ", ", FixedStartMovingEnd, ", ", MovingStartMovingEnd, ", ", time.time() - T0)

# Measure final position
positions.append(agent_pos.copy())

# Process measured data
n = 100
n_large = 2000
goal_data = np.array(goal_data)
avg = np.convolve(goal_data, np.ones(n)/n, mode='valid')
med = []
for i in range(0, goal_data.shape[0]-n):
    arr = goal_data[i:i+n]
    arr = np.sort(arr)
    if n % 2 == 0:
        med_val = arr[int(n/2)]
    else:
        med_val = (arr[int((n-1)/2)] + arr[int((n+1)/2)])/2
    med.append(med_val)
med = np.array(med)

# Plot Average and Median data
plt.plot(avg, label="Average")
plt.plot(med, label="Median")
plt.legend()
plt.show()

# # Plot just average
# plt.plot(avg, label="Average")
# plt.show()
#
# # Plot just median
# plt.plot(med, label="Median")
# plt.show()

# ---------------------------------------------------------------------------------------------------------------

# # Process measured data
# n = 5
# n_large = 4000
# goal_data = np.array(goal_data)
# med = []
# avg = []
# for i in range(0, goal_data.shape[0]-n):
#     arr = goal_data[i:i+n]
#     arr = np.sort(arr)
#     if n % 2 == 0:
#         med_val = arr[int(n/2)]
#     else:
#         med_val = (arr[int((n-1)/2)] + arr[int((n+1)/2)])/2
#     med.append(med_val)
# med = np.array(med)
# for i in range(0, goal_data.shape[0]-n):
#     arr = goal_data[i:i+n]
#     avg.append(np.mean(arr))
# avg = np.array(avg)
#
# steps_to_goal_ittr = np.array(steps_to_goal_ittr)
# steps_to_goal_ittr = np.swapaxes(steps_to_goal_ittr, 0, 1)
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#
# # Plot Average and Median data
# ax1.plot(avg, label="Moving Average")
# ax1.plot(med, label="Moving Median")
# ax1.legend()
# ax1.set_xlabel("Number of Times Goal Reached")
# ax1.set_ylabel("Time steps to Goal")
#
# avg = []
# for i in range(0, len(reward)-n_large):
#     arr = reward[i:i+n_large]
#     avg.append(np.mean(arr))
# avg = np.array(avg)
#
# # Plot Average and Median data
# ax2.plot(avg, label="Average")
# ax2.set_xlabel("Itterations")
# ax2.set_ylabel("Average Reward")
#
# ax3.plot(steps_to_goal_ittr[1], steps_to_goal_ittr[0])
# ax3.set_xlabel("Iteration")
# ax3.set_ylabel("Steps To Goal")
# fig.tight_layout()
# plt.show()

