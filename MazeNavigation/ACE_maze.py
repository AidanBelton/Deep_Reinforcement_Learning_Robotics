import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mazes import Mazes
import matplotlib.pyplot as plt
import time

plt.style.use('science')

class ACE(nn.Module):

    def __init__(self, D_in, learning_rate, lambda_trace=0., discount=0.99):
        super(ACE, self).__init__()
        # Network topology
        self.D_in = D_in
        self.actions = actions
        self.D_out = 1
        self.ReLu = nn.ReLU()
        self.learning_rate = learning_rate

        # Initialize Weights
        self.W = nn.Parameter(torch.zeros(size=(self.D_out, self.D_in)), requires_grad=False)

        self.x_bar = torch.zeros(self.D_in)
        self.y_prev = torch.zeros(self.D_out)
        self.y = torch.zeros(self.D_out)

        self.lambda_trace = lambda_trace
        self.gamma = discount

    def update(self, last_reward, current_sense):

        y = F.linear(current_sense, self.W)
        g = last_reward.reshape(1) + y - self.y_prev

        # Calculate the other product for the batch
        self.x_bar = self.lambda_trace*self.x_bar + (1-self.lambda_trace)*current_sense
        HebbStim = torch.outer(self.x_bar, g)
        OjaNorm = torch.matmul((g*y), self.W)

        self.W += self.learning_rate * (HebbStim.t()-OjaNorm)

        last_value = self.y_prev.clone()
        delta_r = self.gamma*y - self.y_prev

        # update memory and traces
        self.y_prev = y.clone()
        return g, last_value, delta_r


# Define the agent ----------------------------------------------------------------------------------------------------
class Agent:

    def __init__(self, sensory_space, action_space, recent_memory=64, learning_rate=0.001, lambda_trace=0., discount=0.99):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.recent_memory = recent_memory
        self.steps = 0

        # Create networks and optimizer
        self.Hebb_net = ACE(self.sensory_space*self.recent_memory, learning_rate, lambda_trace=lambda_trace, discount=discount)

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.sensory_space*self.recent_memory)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space*self.recent_memory)
        self.last_action = None

    def get_action(self, state):
        with torch.no_grad():
            Y = torch.randint(0, self.action_space, (1, 1))
        return Y

    def learning_step(self, sensors, last_reward):
        last_reward = torch.from_numpy(last_reward).type(torch.FloatTensor)

        # Convert inputs to torch
        sensors = torch.from_numpy(sensors*2-1).type(torch.FloatTensor)

        # update sliding memory
        self.sliding_memory[0:self.sensory_space*(self.recent_memory-1)] = \
            self.sliding_memory.clone()[self.sensory_space:self.sensory_space*self.recent_memory]
        self.sliding_memory[self.sensory_space * (self.recent_memory - 1):self.sensory_space * self.recent_memory] = \
            sensors

        # Get the policies action
        last_internal_reward, last_value, delta_r = self.Hebb_net.update(last_reward, self.sliding_memory)
        action = self.get_action(self.sliding_memory)

        # Update last state and action for state-action-reward-state tuple
        self.last_state = self.sliding_memory.clone()
        self.last_action = action

        self.steps += 1

        return action, last_value, delta_r

# Define the environment ----------------------------------------------------------------------------------------------

# list of possible actions
# [up, down, right, left]
actions = [0, 1, 2, 3]

# Choose the maze
myMazes = Mazes()
M, S, G = myMazes.get_maze(6)
initial_pos = S[0]
goal_pos = G[0]
print(initial_pos, goal_pos)

# Environment settings
max_run_time = 100000
Reward = 0

# Initial Agent Conditions
agent_pos = initial_pos.copy()
old_agent_pos = [1, 1]
old_old_agent_pos = [1, 1]

lr = 0.0001
lambda_trace = 0.1
discount = 0.95
maze_agent = Agent(8, 4, learning_rate=lr, recent_memory=12, lambda_trace=lambda_trace, discount=discount)

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
steps_to_goal_ittr = []
self_action_count = 0
reward = []
delta_r_map = np.zeros(M.shape)
value_map = np.zeros(M.shape)
N_map = np.zeros(M.shape)

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
    agent_action, value, delta_r = maze_agent.learning_step(senses, np.array(Reward))

    delta_r_map[old_agent_pos[0], old_agent_pos[1]] += delta_r
    value_map[old_agent_pos[0], old_agent_pos[1]] += value
    N_map[old_agent_pos[0], old_agent_pos[1]] += 1

    # Store old agents position
    old_old_agent_pos = old_agent_pos.copy()
    old_agent_pos = agent_pos.copy()

    if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
        Reward = 1
        goal_data.append(steps_to_goal)
        n_goals += 1
        print("GOAL!!", steps_to_goal, n_goals, goal_pos)
        steps_to_goal_ittr.append([steps_to_goal, ittr])
        steps_to_goal = 0
        agent_pos = initial_pos.copy()
    else:
        Reward = 0

    # update position if there is no wall
    if agent_action == 0 and M[agent_pos[0] - 1, agent_pos[1]] == 1:  # move up
        agent_pos[0] -= 1
    elif agent_action == 1 and M[agent_pos[0] + 1, agent_pos[1]] == 1:  # move down
        agent_pos[0] += 1
    elif agent_action == 2 and M[agent_pos[0], agent_pos[1] + 1] == 1:  # move right
        agent_pos[1] += 1
    elif agent_action == 3 and M[agent_pos[0], agent_pos[1] - 1] == 1:  # move left
        agent_pos[1] -= 1

    reward.append(Reward)
    steps_to_goal += 1

print("Fixed Start Fixed End", n_goals)
FixedStartFixedEnd = n_goals

values = value_map[M > 0]/(N_map[M > 0]+1e-5)
max_val = np.max(values)
min_val = np.min(values)
print(max_val, min_val)
value_map[M == 0] = min_val-(max_val-min_val)*0.05
plt.imshow(value_map/(N_map+1), vmax=max_val, vmin=min_val-(max_val-min_val)*0.05)
plt.axis('off')
plt.colorbar()
plt.show()


# plt.imshow(delta_r_map/(N_map+1))
# plt.axis('off')
# plt.show()
