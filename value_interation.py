import numpy as np
import random
from scipy.stats import poisson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time  # 引入time模块

# 常数定义
LAMBDA1 = 3  # 第一个地点的租车泊松分布的λ
LAMBDA2 = 4  # 第二个地点的租车泊松分布的λ
SIGMA1 = 3    # 第一个地点的还车泊松分布的σ
SIGMA2 = 2    # 第二个地点的还车泊松分布的σ
MAX_CARS = 20  # 每个地点最多20辆车
MAX_MOVE = 5   # 每天最多移动5辆车
RENTAL_INCOME = 10  # 每辆车租出的收入
MOVE_COST = 2  # 每辆车移动的成本
DISCOUNT = 0.9  # 折扣因子
EPSILON = 1e-4  # 收敛阈值
NUM_ITER = 1000  # 最大迭代次数

# 状态空间：地点1和地点2的车辆数量
states = [(i, j) for i in range(MAX_CARS + 1) for j in range(MAX_CARS + 1)]

# 动作空间：夜间移动的车辆数，范围是[-5, 5]
actions = range(-MAX_MOVE, MAX_MOVE + 1)

# 初始化价值函数和策略
V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)

# 定义一个函数来根据泊松分布生成租车和还车的数量
def rental_and_return_count(lam, sigma):
    rental = poisson.rvs(lam)
    return_ = poisson.rvs(sigma)
    return rental, return_

# 定义奖励函数
def get_reward(state, action, next_state):
    loc1_cars, loc2_cars = state
    next_loc1_cars, next_loc2_cars = next_state

    # 租车收入
    rental_income = 0
    if loc1_cars > 0:  # 地点1有车出租
        rental_income += min(loc1_cars, poisson.rvs(LAMBDA1)) * RENTAL_INCOME
    if loc2_cars > 0:  # 地点2有车出租
        rental_income += min(loc2_cars, poisson.rvs(LAMBDA2)) * RENTAL_INCOME

    # 还车收入（已经还的车数不增加车辆）
    loc1_return = poisson.rvs(SIGMA1)
    loc2_return = poisson.rvs(SIGMA2)

    # 车辆移动的成本
    move_cost = abs(action) * MOVE_COST

    # 奖励 = 租车收入 - 移动车辆的成本
    return rental_income - move_cost

# 定义转移函数
def get_next_state(state, action):
    loc1_cars, loc2_cars = state
    rental_loc1, return_loc1 = rental_and_return_count(LAMBDA1, SIGMA1)
    rental_loc2, return_loc2 = rental_and_return_count(LAMBDA2, SIGMA2)

    # 车辆移动：根据动作调整车辆数量
    loc1_after_move = max(0, loc1_cars - action)  # 将车辆从第一个地点移动到第二个地点
    loc2_after_move = min(MAX_CARS, loc2_cars + action)  # 将车辆从第二个地点移动到第一个地点

    # 计算新状态，确保新状态中的车辆数在合法范围内
    new_loc1_cars = max(0, min(MAX_CARS, loc1_after_move - rental_loc1 + return_loc1))
    new_loc2_cars = max(0, min(MAX_CARS, loc2_after_move - rental_loc2 + return_loc2))

    return new_loc1_cars, new_loc2_cars

# 值迭代算法
def value_iteration():
    global V
    iteration = 0
    while True:
        delta = 0  # 初始化每次迭代的最大值变化
        V_new = np.copy(V)  # 复制当前价值函数
        for state in states:
            best_action_value = float('-inf')
            for action in actions:
                next_state = get_next_state(state, action)
                reward = get_reward(state, action, next_state)
                next_state_value = reward + DISCOUNT * V[next_state]
                best_action_value = max(best_action_value, next_state_value)
            loc1_cars, loc2_cars = state
            delta = max(delta, abs(V[loc1_cars, loc2_cars] - best_action_value))  # 更新最大变化值
            V_new[loc1_cars, loc2_cars] = best_action_value
        V[:] = V_new  # 更新价值函数
        iteration += 1
        if delta < EPSILON or iteration >= NUM_ITER:  # 判断是否收敛
            break
    return V

# 计时开始
start_time = time.time()

# 执行值迭代
V_final = value_iteration()
print("Optimal Policy:")
print(policy)

# 计时结束
end_time = time.time()
execution_time = end_time - start_time  # 计算总耗时

print(f"值迭代算法执行时间: {execution_time:.4f} 秒")

# 可视化：3D图像显示价值函数
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取x, y, z坐标
x = np.array([state[0] for state in states])
y = np.array([state[1] for state in states])
z = np.array([V[state] for state in states])

# 画出3D曲面图
ax.scatter(x, y, z, c=z, cmap='viridis')
ax.set_xlabel('Location 2 Cars')
ax.set_ylabel('Location 1 Cars')
ax.set_zlabel('Value Function')
ax.set_title('Value Function')

plt.show()

# 策略可视化：更新策略（选择最佳行动）
for state in states:
    loc1_cars, loc2_cars = state
    best_action_value = float('-inf')
    best_action = None
    for action in actions:
        next_state = get_next_state(state, action)
        reward = get_reward(state, action, next_state)
        next_state_value = reward + DISCOUNT * V[next_state]
        if next_state_value > best_action_value:
            best_action_value = next_state_value
            best_action = action
    policy[loc1_cars, loc2_cars] = best_action

# 可视化：策略的二维图像
policy_grid = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
for state in states:
    loc1_cars, loc2_cars = state
    policy_grid[loc1_cars, loc2_cars] = policy[state]

plt.figure(figsize=(8, 6))
plt.imshow(policy_grid, cmap='coolwarm', origin='lower', extent=[0, MAX_CARS, 0, MAX_CARS])
plt.colorbar(label='Best Action')
plt.xlabel('Location 2 Cars')
plt.ylabel('Location 1 Cars')
plt.title('Optimal Policy')
plt.show()
