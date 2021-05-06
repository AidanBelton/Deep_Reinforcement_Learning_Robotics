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


class ASE(nn.Module):

    def __init__(self, D_in, D_out, lr=0.001, delta=0.90, neural_noise=0.25):
        super(ASE, self).__init__()
        self.lr = lr

        # Hyperparameters
        self.delta = delta
        self.neural_noise = neural_noise

        # Network Topology
        self.D_in = D_in
        self.D_out = D_out

        # Initialize weights and gradient
        self.W1 = nn.Parameter(torch.normal(0., 1., size=(self.D_in, self.D_out)), requires_grad=False)
        self.W1 /= torch.sum(torch.abs(self.W1), 0).repeat(self.D_in).view(self.D_in, self.D_out)  # Normalise the weights such that they sum to 1
        self.dW1 = torch.zeros((self.D_in, self.D_out))

    def forward_pass(self, x, rng_chance):
        # Compute layer 1
        b = torch.normal(0., self.neural_noise, size=(1, self.D_out)).squeeze()
        y = F.linear(x, self.W1.t(), bias=2.*b)

        # Pick maximum output
        if np.random.uniform() < rng_chance:
            idx = torch.randint(0, self.D_out, (1, 1))
        else:
            _, idx = torch.topk(y, 1, dim=0)
        g = torch.zeros(self.D_out)
        g[idx] = 1

        # Calculate the other product for the batch
        HebbStim = torch.outer(x, g)

        # Update trace
        self.dW1 = self.dW1*self.delta + (1-self.delta)*HebbStim
        return g

    def update(self, reward):
        self.W1 += self.lr * self.dW1 * reward
        # print(torch.linalg.norm(self.W1))

# Define the agent ----------------------------------------------------------------------------------------------------
class Agent:

    def __init__(self, sensory_space, action_space, learning_rate, recent_memory=64, momentum=0.90, neural_noise=0.25, lambda_trace=0.9):
        # Create robots parameters, number of sensors and number of actions
        self.sensory_space = sensory_space
        self.action_space = action_space
        self.recent_memory = recent_memory
        self.steps = 0

        # Create neural network hyperparameters
        self.learning_rate = learning_rate
        self.annealing_rate = 1
        self.epsilon = 0.05


        # Create networks and optimizer
        self.ACE = ACE(self.sensory_space*self.recent_memory, learning_rate=learning_rate, lambda_trace=lambda_trace)
        self.ASE = ASE(self.sensory_space*self.recent_memory, self.action_space, lr=self.learning_rate,
                            delta=momentum, neural_noise=neural_noise)

        # Create sliding sensory memory
        self.sliding_memory = torch.zeros(self.sensory_space*self.recent_memory)

        # Create last state and action for state-action-reward-state tuple
        self.last_state = torch.zeros(self.sensory_space*self.recent_memory)
        self.last_action = None

    def get_action(self, state):
        with torch.no_grad():
            rng_chance = max(-self.steps * self.annealing_rate + 1, self.epsilon)
            Y = torch.argmax(self.ASE.forward_pass(state, rng_chance))
        return Y

    def learning_step(self, sensors, last_reward):
        # Convert inputs to torch
        sensors = torch.from_numpy(sensors*2-1).type(torch.FloatTensor)
        last_reward = torch.from_numpy(last_reward).type(torch.FloatTensor)

        # update sliding memory
        self.sliding_memory[0:self.sensory_space*(self.recent_memory-1)] = \
            self.sliding_memory.clone()[self.sensory_space:self.sensory_space*self.recent_memory]
        self.sliding_memory[self.sensory_space * (self.recent_memory - 1):self.sensory_space * self.recent_memory] = \
            sensors

        # Get the policies action, Update ACE, get internal reward and update ASE
        last_internal_reward, last_value, delta_r = self.ACE.update(last_reward, self.sliding_memory)
        self.ASE.update(last_internal_reward)
        action = self.get_action(self.sliding_memory)

        # Update last state and action for state-action-reward-state tuple
        self.last_state = self.sliding_memory.clone()
        self.last_action = action

        self.steps += 1

        return action, last_value, delta_r

# Define the environment ----------------------------------------------------------------------------------------------


for maze_choice in range(1, 2):
    for n_samples in range(1, 11):
        # list of possible actions
        # [up, down, right, left]
        actions = [0, 1, 2, 3]

        # Choose the maze
        myMazes = Mazes()
        M, S, G = myMazes.get_maze(maze_choice)
        initial_pos = S[0]
        goal_pos = G[0]

        # Environment settings
        max_run_time = 100000
        Reward = 0

        # Initial Agent Conditions
        agent_pos = initial_pos.copy()
        old_agent_pos = [0, 0]
        old_old_agent_pos = [0, 0]
        maze_agent = Agent(8, 4, 0.001, recent_memory=32, momentum=0.9, neural_noise=0.5, lambda_trace=0.7)

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
                # print("GOAL!!", steps_to_goal, n_goals, goal_pos)
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
            # print(agent_pos)

        # print("Fixed Start Fixed End", n_goals)
        FixedStartFixedEnd = n_goals

        # max = np.max(np.abs(value_map)/(N_map+1))
        # print(max)
        #
        # plt.imshow(value_map/(N_map+1), vmax=max, vmin=-max)
        # plt.axis('off')
        # plt.colorbar()
        # plt.show()


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
            agent_action, value, delta_r = maze_agent.learning_step(senses, np.array(Reward))

            # Store old agents position
            old_old_agent_pos = old_agent_pos.copy()
            old_agent_pos = agent_pos.copy()

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
            agent_action, value, delta_r = maze_agent.learning_step(senses, np.array(Reward))

            # Store old agents position
            old_old_agent_pos = old_agent_pos.copy()
            old_agent_pos = agent_pos.copy()

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

        MovingStartMovingEnd = n_goals

        print("\n FINAL Results:", "Maze:", maze_choice, FixedStartFixedEnd, ", ",
              FixedStartMovingEnd, ", ", MovingStartMovingEnd, ", ", time.time() - T0)

        # Measure final position
        positions.append(agent_pos.copy())
        steps_to_goal_ittr = np.array(steps_to_goal_ittr)
        np.savetxt("ASE_ACE_" + str(n_samples) + ".csv", steps_to_goal_ittr, delimiter=",")

        # # Plot Average and Median data
        # plt.plot(avg, label="Average")
        # plt.plot(med, label="Median")
        # plt.legend()
        # plt.show()
