import random
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from torch.autograd import Variable
from torch.distributions import Categorical
from mazes import Mazes
import time


# Define the neural network module ------------------------------------------------------------------------------------

class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.0)
        self.ReLU = nn.ReLU()

        # Policy Memory
        self.log_probs = Variable(torch.Tensor())
        self.rewards = []
        self.values = []

        # Define Policy Gradient Network
        self.L1_p = nn.Linear(D_in, D_out)

        # Define Value Network
        self.L1_v = nn.Linear(D_in, 1)

    def forward(self, X):
        # Calculate Policy
        X_p = self.soft(self.L1_p(X))

        # Calculate Value
        X_v = self.L1_v(X)

        return X_p, X_v

# Define the agent ----------------------------------------------------------------------------------------------------

class Agent:

    def __init__(self, sensory_space, action_space, learning_rate,
                 discount=0.99, recent_memory=64, episode_length=1000):
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

        # Create Q-learning networks and optimizer
        self.A2C_net = Net(self.sensory_space*self.recent_memory, action_space)
        self.optimizer = torch.optim.AdamW(self.A2C_net.parameters(), lr=learning_rate)

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.sensory_space*self.recent_memory)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space*self.recent_memory)
        self.last_action = None
        self.last_value = 0.

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
        last_reward = torch.from_numpy(last_reward).type(torch.FloatTensor)

        # Store reward from last action
        if self.steps != 0:
            self.A2C_net.rewards.append(last_reward)

        # update sliding memory
        self.sliding_memory[0:self.sensory_space*(self.recent_memory-1)] = \
            self.sliding_memory.clone()[self.sensory_space:self.sensory_space*self.recent_memory]
        self.sliding_memory[self.sensory_space * (self.recent_memory - 1):self.sensory_space * self.recent_memory] = \
            sensors

        # If the episode has finished
        self.steps += 1
        if self.steps % self.episode_length == 0 and self.steps > 0:
            self.train_policy()

        # Get the policies action
        action, _ = self.get_action(self.sliding_memory)

        return action

# Define the environment ----------------------------------------------------------------------------------------------

for maze_choice in range(1, 6):
    for n_samples in range(0, 15):

        # list of possible actions
        # [up, down, right, left]
        actions = [0, 1, 2, 3]

        # Choose the maze
        myMazes = Mazes()
        M, S, G = myMazes.get_maze(maze_choice)
        initial_pos = S[0]
        goal_pos = G[0]

        # Initial Agent Conditions
        agent_pos = initial_pos.copy()
        old_agent_pos = [0, 0]
        old_old_agent_pos = [0, 0]
        maze_agent = Agent(8, 4, 0.01, discount=0.99, recent_memory=8, episode_length=1000)

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

        # print("Fixed Start Fixed End", n_goals)
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

        # print("Fixed Start Moving End", n_goals)
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

        print("\n FINAL Results:", "Maze:", maze_choice, FixedStartFixedEnd, ", ", FixedStartMovingEnd, ", ", MovingStartMovingEnd, ", ", time.time() - T0)


