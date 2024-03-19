import numpy as np
from bandit import Bandit, Agent

runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps)) # (200, 1000)

for run in range(runs): # 200번 실험
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward/(step+1))

    all_rates[run] = rates  # 보상 결과 기록

avg_rates = all_rates.mean(axis=0) # 200번 실험의 평균 보상 계산
