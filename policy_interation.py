import numpy as np
import random
from scipy.stats import poisson
import matplotlib.pyplot as plt
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

# 策略迭代算法
def policy_iteration():
    global V, policy
    iteration = 0
    while True:
        # 1. 策略评估（计算给定策略下的价值函数）
        delta = 0
        while True:
            delta = 0
            V_new = np.copy(V)
            for state in states:
                loc1_cars, loc2_cars = state
                action = policy[loc1_cars, loc2_cars]  # 根据当前策略选择动作
                next_state = get_next_state(state, action)
                reward = get_reward(state, action, next_state)
                next_state_value = reward + DISCOUNT * V[next_state]
                delta = max(delta, abs(V[loc1_cars, loc2_cars] - next_state_value))
                V_new[loc1_cars, loc2_cars] = next_state_value
            V[:] = V_new
            if delta < EPSILON:
                break

        # 2. 策略改进（更新策略）
        policy_stable = True
        for state in states:
            loc1_cars, loc2_cars = state
            old_action = policy[loc1_cars, loc2_cars]
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
            if old_action != best_action:
                policy_stable = False

        iteration += 1
        if policy_stable:
            break
        if iteration >= NUM_ITER:  # 避免无限循环
            break
    return V, policy

# 计时开始
start_time = time.time()

# 执行策略迭代
V_final, policy_final = policy_iteration()
print("Optimal Policy:")
print(policy_final)

# 计时结束
end_time = time.time()
execution_time = end_time - start_time  # 计算总耗时

print(f"策略迭代算法执行时间: {execution_time:.4f} 秒")

# 可视化：3D图像显示价值函数
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取x, y, z坐标
x = np.array([state[0] for state in states])
y = np.array([state[1] for state in states])
z = np.array([V_final[state] for state in states])

# 画出3D曲面图
ax.scatter(x, y, z, c=z, cmap='viridis')
ax.set_xlabel('Location 1 Cars')
ax.set_ylabel('Location 2 Cars')
ax.set_zlabel('Value Function')
ax.set_title('Value Function')

plt.show()

# 策略可视化：更新策略（选择最佳行动）
policy_grid = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
for state in states:
    loc1_cars, loc2_cars = state
    policy_grid[loc1_cars, loc2_cars] = policy_final[state]

# 可视化：策略的二维图像
plt.figure(figsize=(8, 6))
plt.imshow(policy_grid, cmap='coolwarm', origin='lower', extent=[0, MAX_CARS, 0, MAX_CARS])
plt.colorbar(label='Best Action')
plt.xlabel('Location 1 Cars')
plt.ylabel('Location 2 Cars')
plt.title('Optimal Policy')
plt.show()
