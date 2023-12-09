import numpy as np


class EXP3:
    def __init__(self, num_arms, simulations_num, rewards, actions, gamma=0.1, eta=0.1):
        self.num_arms = num_arms
        self.gamma = gamma
        self.eta = eta
        self.weights = np.ones(num_arms)
        self.simulations_num = simulations_num
        self.reward_table = rewards
        self.actions = actions
        self.probabilities = np.ones(num_arms) / num_arms
        self.best_action = None
        self.current_reward = float('-inf')

    def choose_arm(self):
        return np.random.choice(self.num_arms, p=self.probabilities)

    def update(self, chosen_arm, reward):
        estimated_reward = reward / self.probabilities[chosen_arm]
        self.weights[chosen_arm] *= np.exp(self.eta * estimated_reward)
        normalization_factor = np.sum(self.weights)
        self.probabilities = (1 - self.gamma) * (self.weights /
                                                 normalization_factor) + (self.gamma / self.num_arms)

    def simulate(self):
        time_steps = [i for i in range(self.simulations_num)]
        rewards = []
        for _ in range(self.simulations_num):
            chosen_arm = self.choose_arm()
            action = self.actions[chosen_arm]
            reward = self.reward_table[chosen_arm]
            if self.current_reward < reward:
                self.current_reward = reward
                self.best_action = action
            rewards.append(reward)
            self.update(chosen_arm=chosen_arm, reward=reward)
        return time_steps, rewards

    def get_best_action(self):
        return self.best_action
