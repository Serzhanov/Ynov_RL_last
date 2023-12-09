import numpy as np
ucb_c = 2


class UCB():
    """
    Epsilon Greedy with incremental update.
    Based on Sutton and Barto pseudo-code, page. 24
    """

    def __init__(self, bandit):
        self.current_reward = float('-inf')
        self.best_arm = None
        global ucb_c
        self.ucb_c = ucb_c
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.Q = np.zeros(self.arm_count)  # q-value of actions
        self.N = np.zeros(self.arm_count) + 0.0001  # action count
        self.timestep = 1

    @staticmethod
    def name():
        return 'ucb'

    def get_action(self):
        ln_timestep = np.log(np.full(self.arm_count, self.timestep))
        confidence = self.ucb_c * np.sqrt(ln_timestep/self.N)
        action = np.argmax(self.Q + confidence)
        self.timestep += 1
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
        return self.bandit.reward_table[self.best_arm]
