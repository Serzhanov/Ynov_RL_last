import numpy as np
epsilon = 0.1


class EpsilonGreedy():
    """
    Epsilon Greedy with incremental update.
    Based on Sutton and Barto pseudo-code, page. 24
    """

    def __init__(self, bandit):
        global epsilon
        self.best_arm = None
        self.current_reward = float('-inf')
        self.epsilon = epsilon
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.Q = np.zeros(self.arm_count)  # q-value of actions
        self.N = np.zeros(self.arm_count)  # action count

    @staticmethod
    def name():
        return 'epsilon-greedy'

    def get_action(self):
        if np.random.uniform(0, 1) > self.epsilon:
            action = self.Q.argmax()
        else:
            action = np.random.randint(0, self.arm_count)
        return action

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        if self.current_reward < reward:
            self.current_reward = reward
            self.best_arm = arm
        self._update_params(arm, reward)
        return reward, regret

    def _update_params(self, arm, reward):
        self.N[arm] += 1  # increment action count
        self.Q[arm] += 1/self.N[arm] * \
            (reward - self.Q[arm])  # inc. update rule

    def get_best_action(self):
        return self.bandit.actions[self.best_arm]
