import random
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mazes import Mazes
import time

# Define the neural network module ------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.L1 = nn.Linear(D_in, D_out)

    def forward(self, X):
        X = self.L1(X)
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
                 epsilon, discount=0.99, batch_size=64, recent_memory=64, replay_memory=10000, update_freq=10000):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.recent_memory = recent_memory
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
        self.on_policy_net = Net(self.sensory_space*self.recent_memory, self.action_space)
        self.off_policy_net = Net(self.sensory_space*self.recent_memory, self.action_space)
        self.on_policy_net.load_state_dict(self.on_policy_net.state_dict())
        self.on_policy_net.eval()
        self.optimizer = torch.optim.AdamW(self.off_policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.sensory_space*self.recent_memory)

        # Create memory replay
        self.memory_replay = Replay_Memory(replay_memory, self.sensory_space*self.recent_memory, batch_size=self.batch_size)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space*self.recent_memory)
        self.last_action = None

    def get_action(self, state):
        if np.random.uniform() < max(-self.steps*self.annealing_rate + 1, self.epsilon):
            Y = torch.randint(0, self.action_space, (1, 1))
            epsilon = True
        else:
            self.on_policy_net.eval()
            with torch.no_grad():
                state = state.unsqueeze(0)
                Y = torch.argmax(self.on_policy_net(state))
            epsilon = False
        return Y, epsilon

    def train(self):
        b_states, b_actions, b_rewards, b_futr_states = self.memory_replay.get_batch()
        b_actions = b_actions.type(torch.LongTensor)

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

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch
        sensors = torch.from_numpy(sensors).type(torch.FloatTensor)
        last_reward = torch.from_numpy(last_reward).type(torch.FloatTensor)

        # update sliding memory
        self.sliding_memory[0:self.sensory_space*(self.recent_memory-1)] = \
            self.sliding_memory.clone()[self.sensory_space:self.sensory_space*self.recent_memory]
        self.sliding_memory[self.sensory_space * (self.recent_memory - 1):self.sensory_space * self.recent_memory] = \
            sensors

        # Store state-action-reward-state tuple in memory replay
        if self.last_action is not None:
            self.memory_replay.push(self.last_state, self.last_action, last_reward, self.sliding_memory)

        # If enough data train the off policy network
        if len(self.memory_replay) > self.batch_size:
            self.train()

        # Get the policies action
        action, epsilon = self.get_action(self.sliding_memory)

        # Update last state and action for state-action-reward-state tuple
        self.last_state = self.sliding_memory.clone()
        self.last_action = action

        self.steps += 1
        if self.steps % self.update_freq == 0:
            self.on_policy_net.load_state_dict(self.off_policy_net.state_dict())
            self.on_policy_net.eval()

        return action, epsilon

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

# Initial Agent Conditions
agent_pos = initial_pos.copy()
old_agent_pos = [0, 0]
old_old_agent_pos = [0, 0]
maze_agent = Agent(8, 4, 0.01, 1/100000, .05,
                   discount=0.99, batch_size=64, recent_memory=20, replay_memory=10000, update_freq=10000)

# Environment settings
max_run_time = 100000
Reward = 0

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
    agent_action, not_self_action = maze_agent.learning_step(senses, np.array(Reward))

    if not not_self_action:
        self_action_count += 1

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

    if old_agent_pos == agent_pos.copy() and not not_self_action:
        run_into_wall.append(1)
    else:
        run_into_wall.append(0)

    if old_old_agent_pos == agent_pos.copy() and not not_self_action:
        wasted_step.append(1)
    else:
        wasted_step.append(0)

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
    agent_action, not_self_action = maze_agent.learning_step(senses, np.array(Reward))

    if not not_self_action:
        self_action_count += 1

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

    if old_agent_pos == agent_pos.copy() and not not_self_action:
        run_into_wall.append(1)
    else:
        run_into_wall.append(0)

    if old_old_agent_pos == agent_pos.copy() and not not_self_action:
        wasted_step.append(1)
    else:
        wasted_step.append(0)

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
    agent_action, not_self_action = maze_agent.learning_step(senses, np.array(Reward))

    if not not_self_action:
        self_action_count += 1

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

    if old_agent_pos == agent_pos.copy() and not not_self_action:
        run_into_wall.append(1)
    else:
        run_into_wall.append(0)

    if old_old_agent_pos == agent_pos.copy() and not not_self_action:
        wasted_step.append(1)
    else:
        wasted_step.append(0)

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
avg_run_into_wall = np.convolve(np.array(run_into_wall), np.ones(n_large)/n_large)
avg_wasted_step = np.convolve(np.array(wasted_step), np.ones(n_large)/n_large)
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

print("Number of non epsilon actions:", self_action_count)

# Plot running into walls and wasted steps
plt.plot(avg_run_into_wall, label="Runing into wall")
plt.plot(avg_wasted_step, label="Wasted step")
plt.legend()
plt.show()
#
# # Plot Average and Median data
# plt.plot(avg, label="Average")
# plt.plot(med, label="Median")
# plt.legend()
# plt.show()
#
# # Plot just average
# plt.plot(avg, label="Average")
# plt.show()
#
# # Plot just median
# plt.plot(med, label="Median")
# plt.show()

# ---------------------------------------------------------------------------------------------------------------

# Process measured data
n = 5
n_large = 4000
goal_data = np.array(goal_data)
med = []
avg = []
for i in range(0, goal_data.shape[0]-n):
    arr = goal_data[i:i+n]
    arr = np.sort(arr)
    if n % 2 == 0:
        med_val = arr[int(n/2)]
    else:
        med_val = (arr[int((n-1)/2)] + arr[int((n+1)/2)])/2
    med.append(med_val)
med = np.array(med)
for i in range(0, goal_data.shape[0]-n):
    arr = goal_data[i:i+n]
    avg.append(np.mean(arr))
avg = np.array(avg)

steps_to_goal_ittr = np.array(steps_to_goal_ittr)
steps_to_goal_ittr = np.swapaxes(steps_to_goal_ittr, 0, 1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Plot Average and Median data
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



# Animation function --------------------------------------------------------------------------------------------------
# def animate(i):
#     Map.set_data(positions[i, 1], positions[i, 0])
#     ax.set_title("Iteration: " + str(i))
#     return Map,
#
# positions = np.array(positions)
# positions[:, 0] = M.shape[0]-1 - positions[:, 0]
# print(positions)
# fig, ax = plt.subplots()
# Map, = ax.plot(positions[0, 0], positions[0, 1], 'go')
# colors = ['red']
#
# M = np.flip(M, 0)
# M_scatter_vals = []
# for i in range(0, M.shape[0]):
#     for j in range(0, M.shape[1]):
#         if M[i][j] == 0:
#             M_scatter_vals.append([i, j])
# M_scatter_vals = np.array(M_scatter_vals)
#
# plt.scatter(M_scatter_vals[:, 1], M_scatter_vals[:, 0])
# ax.set(xlim=(-1, M.shape[1]+2), ylim=(-1, M.shape[0]+2))
# plt.axis('off')
#
# myAnimation = animation.FuncAnimation(fig, animate, frames=range(1, len(positions)), interval=20,
#                                       blit=True, repeat=True)
# plt.show()
#
# writer = animation.PillowWriter(fps=60)
# writer.setup(fig, 'changing_weights.gif')
# myAnimation.save('changing_weights.gif', writer=writer)

