import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from game import Game2048
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

input_size = 16  # Size of the input state
output_size = 4  # Number of possible actions (up, down, left, right)
batch_size = 64  # Size of the training batch
gamma = 0.99  # Discount factor for future rewards
epsilon = 0.9  # Exploration rate
epsilon_decay = 0.9999  # Decay rate for exploration rate
epsilon_min = 0.01  # Minimum exploration rate END
target_update = 20  # Number of episodes to update the target network
num_episodes = 5000  # Number of episodes to train

total_rewards = []
episodes = []
steps_done = 0

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_size, output_size, batch_size, gamma, target_update):
        global steps_done
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.policy_net.load_state_dict(torch.load('policy_net.pth'))
        # self.target_net.load_state_dict(torch.load('target_net.pth'))
        self.target_net.eval()

        self.policy_net.train()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(10000) # TODO 50000

        steps_done = 0

    def select_action(self, state, epsilon):
        global steps_done
        steps_done += 1
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
        
    def update_model(self):
        global steps_done
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)

        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(Transition(state, action, reward, next_state, done))

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "policy_net.pth")
        torch.save(self.target_net.state_dict(), "target_net.pth")

    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.eval()


agent = DQNAgent(input_size, output_size, batch_size, gamma, target_update)

total_scores, best_tile_list = [], []

for episode in range(1, num_episodes + 1):
    game = Game2048()
    state = np.array(game.matrix).flatten()
    done = False
    total_reward = 0
    non_valid_count, valid_count = 0, 0

    while not done:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        action = agent.select_action(state, epsilon)
        
        old_merge_score = game.get_merge_score()
        old_max_num = game.max_num()
        old_sum = game.get_sum()

        game.make_move(action)

        next_state = np.array(game.matrix).flatten()
        reward = (game.get_merge_score() - old_merge_score + game.max_num() - old_max_num + game.get_sum() - old_sum)
        total_reward += reward
        # reward = torch.tensor([reward], device=agent.device)

        done = game.game_end

        if np.array_equal(state, next_state):
            reward -= 20 
            non_valid_count += 1
        else:
            valid_count += 1

        agent.store_transition(state, action, reward, next_state, done)

        state = next_state        

        if done:
            agent.update_model()
            
            print(f"Score: {game.get_merge_score()}")
            print(f"Last board sum: {game.get_sum()}")
            print(f"Max num: {game.max_num()}")
            total_scores.append(game.get_merge_score())
            best_tile_list.append(game.max_num())
            break
        

    total_rewards.append(total_reward)
    episodes.append(episode)


    if episode % target_update == 0:
        agent.update_target_network()

    print(f"Episode: {episode}, Total Reward: {total_reward}")
    if episode % 100 == 0:
        agent.save_model()
    

plt.plot(episodes, total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episode')
plt.show()

plt.plot(total_scores, best_tile_list)
plt.xlabel('Total score')
plt.ylabel('Best tile')
plt.title('Total score vs Best tile')
plt.show()
