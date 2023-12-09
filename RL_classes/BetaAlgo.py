import numpy as np


class BetaAlgo():
    """
    The algos try to learn which Bandit arm is the best to maximize reward.

    It does this by modelling the distribution of the Bandit arms with a Beta, 
    assuming the true probability of success of an arm is Bernouilli distributed.
    """

    def __init__(self, bandit):
        """
        Args:
          bandit: the bandit class the algo is trying to model
        """
        self.best_arm = None
        self.current_reward = float('-inf')
        self.bandit = bandit
        self.arm_count = bandit.arm_count
        self.alpha = np.ones(self.arm_count)
        self.beta = np.ones(self.arm_count)

    def get_reward_regret(self, arm):
        reward, regret = self.bandit.get_reward_regret(arm)
        if self.current_reward < reward:
            self.current_reward = reward
            self.best_arm = arm
        self._update_params(arm, reward)
        return reward, regret

    def _update_params(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        if (self.alpha <= 0).any() or (self.beta <= 0).any():
            self.alpha = np.where(self.alpha <= 0, 0.0001, self.alpha)
            self.beta = np.where(self.beta <= 0, 0.0001, self.beta)


class BernGreedy(BetaAlgo):
    def __init__(self, bandit):
        super().__init__(bandit)

    @staticmethod
    def name():
        return 'beta-greedy'

    def get_action(self):
        """ Bernouilli parameters are the expected values of the beta"""
        theta = self.alpha / (self.alpha + self.beta)
        return theta.argmax()


class BernThompson(BetaAlgo):
    def __init__(self, bandit):
        self.best_arm = None
        super().__init__(bandit)

    @staticmethod
    def name():
        return 'thompson'

    def get_action(self):
        """ Bernouilli parameters are sampled from the beta"""
        theta = np.random.beta(self.alpha, self.beta)
        return theta.argmax()

    def get_best_action(self):
        return self.bandit.actions[self.best_arm]
