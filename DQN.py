import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random
import numpy as np
import itertools
import seaborn as sns

import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, action_values):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.action_heads = nn.ModuleList([nn.Linear(256, action_values) for _ in range(action_dim)])
        self.reset_parameters()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = torch.stack([head(x) for head in self.action_heads], dim=1)  # (batch, action_dim, action_values)即(batch_size, 9, 7) 
        return q_values # 9个动作各7个取值的价值

    def reset_parameters(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for head in self.action_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

n_locations = 10 # number of locations
days = 20

rent_means=np.random.randint(1, 6, n_locations)
return_means=np.random.randint(1, 6, n_locations)
car_num=4*np.ones(n_locations) # suppose each location has initial 4 cars
car_max=8
car_max_move=3
rent_credict=10
move_car_cost=2

# combinations = list(itertools.combinations(range(0, n_locations), 2))
# actions_indexs = [list(comb) for comb in combinations] 

# 获取状态和动作的维度
state_dim = n_locations
action_dim = n_locations-1  # 10个地点间相互移动的动作维度9
action_values = 2*car_max_move+1 # 每个动作有 7 个选择

deltas=return_means-rent_means
location_move_car_proiority=np.argsort(deltas)[::-1] # 优先移动车辆的地点

# 创建 Q 网络和目标网络
q_network = DQN(state_dim, action_dim,action_values)  # 不断更新
target_network = DQN(state_dim, action_dim,action_values)  # C 步后更新
target_network.load_state_dict(q_network.state_dict())

# 定义优化器
optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

print('rent_means:',rent_means,'return_means:',return_means)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(capacity=50000)



def epsilon_greedy(q_values, epsilon):
    """
    ε-贪心策略选择动作。

    参数:
    - q_values: numpy 数组，形状为 (action_dim, action_values)，每个维度的 Q 值。
    - epsilon: float 探索概率。

    返回:
    - actions: numpy 数组，形状为 (action_dim,)。
    """
    action_dim, action_values = q_values.shape
    actions = []
    for dim in range(action_dim):
        if np.random.rand() < epsilon:
            # 随机选择动作
            actions.append(np.random.randint(0, action_values))
        else:
            # 选择 Q 值最大的动作
            actions.append(np.argmax(q_values[dim]))

    return np.array(actions)  # Shape: (action_dim,)

# 环境步进
def step(state, actions):
    statec = np.copy(state)
    rewards = np.zeros(action_dim)

    # # 还需考虑环境变量 
    rents=np.random.poisson(rent_means)
    returns=np.random.poisson(return_means)
    
    for i in location_move_car_proiority: # 优先移动车辆的地点
        if i == action_dim: continue
        reward_i=0

        # action of movement 
        move_car=actions[i]
        statec[i] = max(0, min(car_max, statec[i] - move_car))
        statec[i+1] = max(0, min(car_max, state[i+1] + move_car))
        reward_i -= move_car_cost * abs(move_car)

        # environment of rent 
        real_rent1=min(statec[i],rents[i])
        real_rent2=min(statec[i+1],rents[i+1])
        statec[i]-=real_rent1
        statec[i+1]-=real_rent2

        # environment of return
        statec[i]+=returns[i]
        statec[i+1]+=returns[i+1]

        statec[i]=min(car_max,statec[i])
        statec[i+1]=min(car_max,statec[i+1])

        
        if i != action_dim-1:
            reward_i+=real_rent1*rent_credict
            rewards[i]=reward_i
        else: # add the last location reward
            reward_i+=real_rent1*rent_credict+real_rent2*rent_credict
            rewards[i]=reward_i

    return statec, rewards


def train(batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    # 从重放缓冲区中采样
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    actions = np.array(actions) + car_max_move  # 将每个动作值加 car_max_move，使其在 [0, 2*car_max_move] 范围内。(还原操作)

    # 转换为张量
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)  # Shape: (batch_size, action_dim) 
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # Shape: (64,9)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)

    # 计算当前 Q 值
    q_values = q_network(states)  # (64,9,7)
    # print(q_values.shape)
    # 从 Q 值中选择每个动作维度的对应动作
    chosen_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (64,9)
    # print(chosen_q_values.shape)

    # 计算下一个状态的最大 Q 值
    with torch.no_grad():
        next_q_values = target_network(next_states)  # Shape: (batch_size, action_dim, action_values)
        max_next_q_values = next_q_values.max(dim=2)[0]  # (64,9)

    # print(rewards.shape)
    # print(max_next_q_values.shape)
    # 计算目标 Q 值
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)  # Shape: (batch_size, action_dim)

    # print(target_q_values)
    # print(chosen_q_values)

    # 计算损失
    loss = nn.MSELoss()(chosen_q_values, target_q_values)

    # 更新网络
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# 训练参数
num_episodes = 500
batch_size = 64
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.02
epsilon_decay = 500
decay=0.9985
target_update_freq = 3

total_steps = 0
episode_rewards = []
losses = []

epsilon = epsilon_start
for episode in range(1,num_episodes+1):
    loss_episode=0
    state = car_num
    done = False
    episode_reward = 0
    # while not done:
    for i in range(days):
        with torch.no_grad():
            q_values = q_network(torch.tensor(state,dtype=torch.float32))
        # print(q_values.shape)
        q_values = q_values.numpy().transpose(1, 0)
        action = epsilon_greedy(q_values, epsilon) - car_max_move # 由于动作空间是 [-3, 3]，所以需要减去 3
        # print(action)
        # action_rewards = [location1_reward, location2_reward, ..., location9_reward + location10_reward] action和每个state reward是一一对应的
        next_state, action_rewards = step(state,action)

        # print(reward)
        if i==days-1: done=True
        replay_buffer.add(state, action, action_rewards, next_state, done)
        state = next_state

        if total_steps > 500:
            loss=train(batch_size, gamma)
            loss_episode+=loss
       
        # 更新 ε
        # epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
        epsilon = max(epsilon_end,epsilon*decay)

        total_steps += 1
        episode_reward += np.sum(action_rewards)

    episode_rewards.append(episode_reward)

    if total_steps > 500:
        losses.append(loss_episode / days)
        
    # 定期更新目标网络
    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    # 打印进度
    if episode % 10 == 0:
         if total_steps > 500:
            print(f'Episode {episode}, Epsilon: {epsilon:.2f}, Episode reward: {episode_reward:.2f}, Loss: {loss_episode/days:.4f}')
         else:
            print(f'Episode {episode}, Epsilon: {epsilon:.2f}, Episode reward: {episode_reward:.2f}')
    
# episode_rewards可视化
plt.figure()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.title('Episode rewards over time')
plt.show()

# losses可视化
plt.figure()
plt.plot(losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss over time')

plt.show()

# 展示show_state状态输入至网络后的最优策略可视化
show_state=np.random.randint(0, car_max + 1, size=n_locations)
# 获取最优策略
def get_optimal_policy(state):
    with torch.no_grad():
        q_values = q_network(torch.tensor(state, dtype=torch.float32))  # Shape: (action_dim, action_values)
        q_values = q_values.numpy().transpose(1, 0)  # Shape: (action_values, action_dim) 
        actions = []
        for dim in range(action_dim):
            # 选择 Q 值最大的动作
            actions.append(np.argmax(q_values[dim]))
    return np.array(actions)  # Shape: (action_dim,)


def visualize_policy_heatmap(state):
    optimal_actions = get_optimal_policy(state)
    heatmap_matrix = np.full((n_locations - 1, n_locations), np.nan)  # 初始化为 NaN

    # 填充热力图矩阵
    for src in range(action_dim):
        dest = src + 1  # Destination location (x-coordinate)
        heatmap_matrix[src, dest] = optimal_actions[src] - car_max_move  # Adjust action range to [-car_max_move, car_max_move]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True, cbar_kws={'label': 'Optimal Move (Cars)'})
    plt.xlabel('Destination Location')
    plt.ylabel('Source Location')
    plt.title('Optimal Policy Heatmap in state {}'.format(state))
    plt.tight_layout()
    plt.show()
    

# 展示 show_state 的最优策略热力图
visualize_policy_heatmap(show_state)
plt.show()






