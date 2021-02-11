import random
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

# Define the neural network module ------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, D_in, seq_len, D_out):
        super(Net, self).__init__()
        self.seq_len = seq_len

        self.LSTM = nn.LSTM(D_in, 50, 1, batch_first=True)
        self.L1 = nn.Linear(50*seq_len, D_out)

    def forward(self, X):
        X,_ = self.LSTM(X)
        X = X.reshape(X.shape[0], -1)
        X = self.L1(X)
        return X

# Define replay memory ------------------------------------------------------------------------------------------------
class Replay_Memory:

    def __init__(self, maximum_size, seq_len, state_space, batch_size=64):
        self.maximum_size = maximum_size
        self.state_space = state_space
        self.write_head = 0
        self.fully_written = False
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Define memory blocks
        self.states = torch.zeros((self.maximum_size, self.seq_len, state_space))
        self.actions = torch.zeros((self.maximum_size, 1))
        self.rewards = torch.zeros((self.maximum_size, 1))
        self.future_states = torch.zeros((self.maximum_size, self.seq_len, state_space))

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

    def __init__(self, sensory_space, action_space, recent_memory=64):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.recent_memory = recent_memory
        self.steps = 0

        # Create the MDP's hyperparameters
        self.discount = 0.99
        self.annealing_rate = 1/10000
        self.epsilon = 0.05

        # Create neural network hyperparameters
        self.batch_size = 128
        self.update_freq = 10000
        self.learning_rate = 0.001

        # Create networks and optimizer
        self.on_policy_net = Net(self.sensory_space, self.recent_memory, self.action_space)
        self.off_policy_net = Net(self.sensory_space, self.recent_memory, self.action_space)
        self.on_policy_net.load_state_dict(self.on_policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.off_policy_net.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss()

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.recent_memory, self.sensory_space)

        # Create memory replay
        self.memory_replay = Replay_Memory(10000, self.recent_memory, self.sensory_space, batch_size=self.batch_size)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.recent_memory, self.sensory_space)
        self.last_action = None

    def get_action(self, state):
        if np.random.uniform() < (1-self.epsilon)*np.exp(-self.steps)+self.epsilon:
            Y = torch.randint(0, self.action_space, (1, 1))
        else:
            with torch.no_grad():
                Y = torch.argmax(self.on_policy_net(state))
        return Y

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
        self.sliding_memory[0:(self.recent_memory-1)] = self.sliding_memory.clone()[1:self.recent_memory]
        self.sliding_memory[(self.recent_memory - 1):self.recent_memory] = sensors

        # Store state-action-reward-state tuple in memory replay
        if self.last_action is not None:
            self.memory_replay.push(self.last_state, self.last_action, last_reward, self.sliding_memory)

        # If enough data train the off policy network
        if len(self.memory_replay) > self.batch_size:
            self.train()

        # Get the policies action
        action = self.get_action(self.sliding_memory.reshape(1, self.recent_memory, -1))

        # Update last state and action for state-action-reward-state tuple
        self.last_state = self.sliding_memory.clone()
        self.last_action = action

        self.steps += 1
        if self.steps % self.update_freq:
            self.on_policy_net.load_state_dict(self.off_policy_net.state_dict())

        return action

# Define the environment ----------------------------------------------------------------------------------------------

# list of possible actions
# [up, down, right, left]
actions = [0, 1, 2, 3]

# Maze, 0 is a wall 1 is an open space
M = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
              [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Initial Agent Conditions
initial_pos = [1, 1]
agent_pos = initial_pos.copy()
maze_agent = Agent(4, 4)

# Environment settings
max_run_time = 2000
goal_pos = [8, 13]
switch_pos = [6, 7]
flipped_switch = False
Reward = 0

# Data collection
steps_to_goal = 0
n_goals = 0
goal_data = []
positions = []

for ittr in range(0, max_run_time):
    positions.append(agent_pos.copy())

    # Get the agents sensory information
    senses = []
    senses.append(M[agent_pos[0]-1, agent_pos[1]])
    senses.append(M[agent_pos[0]+1, agent_pos[1]])
    senses.append(M[agent_pos[0], agent_pos[1]-1])
    senses.append(M[agent_pos[0], agent_pos[1]+1])
    senses = np.array(senses)

    # Get the agents actions
    agent_action = maze_agent.learning_step(senses, np.array(Reward))

    # update position if there is no wall
    if agent_action == 0 and M[agent_pos[0] - 1, agent_pos[1]] == 1:  # move up
        agent_pos[0] -= 1
    elif agent_action == 1 and M[agent_pos[0] + 1, agent_pos[1]] == 1:  # move down
        agent_pos[0] += 1
    elif agent_action == 2 and M[agent_pos[0], agent_pos[1] + 1] == 1:  # move right
        agent_pos[1] += 1
    elif agent_action == 3 and M[agent_pos[0], agent_pos[1] - 1] == 1:  # move left
        agent_pos[1] -= 1

    if agent_pos[0] == switch_pos[0] and agent_pos[1] == goal_pos[1]:
        flipped_switch = True
    if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1] and flipped_switch:
        Reward = 1
        goal_data.append(steps_to_goal)
        n_goals += 1
        print("GOAL!!", steps_to_goal, n_goals)
        steps_to_goal = 0
        flipped_switch = False
    else:
        Reward = 0
    print(agent_pos)
    steps_to_goal += 1

# Measure final position
positions.append(agent_pos.copy())

# n = 100
# goal_data = np.array(goal_data)
# avg = np.convolve(goal_data, np.ones(n)/n, mode='valid')
# # plt.plot(goal_data)
# plt.plot(avg)
# plt.show()


# Animation function --------------------------------------------------------------------------------------------------
def animate(i):
    Map.set_data(positions[i, 1], positions[i, 0])
    ax.set_title("Iteration: " + str(i))
    return Map,

positions = np.array(positions)
positions[:, 0] = 11 - positions[:, 0]
print(positions)
fig, ax = plt.subplots()
Map, = ax.plot(positions[0, 0], positions[0, 1], 'go')
colors = ['red']

M = np.flip(M, 0)
M_scatter_vals = []
for i in range(0, M.shape[0]):
    for j in range(0, M.shape[1]):
        if M[i][j] == 0:
            M_scatter_vals.append([i, j])
M_scatter_vals = np.array(M_scatter_vals)

plt.scatter(M_scatter_vals[:, 1], M_scatter_vals[:, 0])
ax.set(xlim=(-1, 17), ylim=(-1, 12))
plt.axis('off')

myAnimation = animation.FuncAnimation(fig, animate, frames=range(1, len(positions)), interval=20,
                                      blit=True, repeat=True)
plt.show()

writer = animation.PillowWriter(fps=60)
writer.setup(fig, 'changing_weights.gif')
myAnimation.save('changing_weights.gif', writer=writer)

