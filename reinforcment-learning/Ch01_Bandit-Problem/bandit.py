import numpy as np

class Bandit:
    def __init__(self, arms=10): # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms) # 슬롯머신 각각의 승률 설정(무작위)
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self, epsilon, actgion_size=10) -> None:
        self.epsilon = epsilon # 무작위로 행동할 확률(탐ㅅ핵 확률)
        self.Qs = np.zeros(actgion_size)
        self.ns = np.zeros(actgion_size)

    def update(self, action, reward): # 슬롯머신의 가치 추정
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self): # 행동 선택(epsilon-greedy policy)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        return np.argmax(self.Qs) # 탐욕 행동 선택
    
if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = [] # 시간에 따른 누적 보상 저장(보상 합)
    rates = [] # 시간에 따른 슬롯머신의 승률 저장

    for step in range(steps):
        action = agent.get_action()  # 1) 행동 선택
        reward = bandit.play(action) # 2) 실제로 플레이하고 보상 획득
        agent.update(action, reward) # 3) 행동과 보상을 통해 학습
        total_reward += reward
        total_rewards.append(total_reward)  #) 현재까지의 보상 합 저장
        rates.append(bandit.rates)          #) 현재까지의 슬롯머신 승률 저장

    print(total_reward)