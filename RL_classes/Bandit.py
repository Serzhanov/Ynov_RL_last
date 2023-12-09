import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

stationary = True


class Bandit():
    def __init__(self, arm_count, actions, rewards):
        """
        Multi-armed bandit with rewards 1 or 0.

        At initialization, multiple arms are created. The probability of each arm
        returning reward 1 if pulled is sampled from Bernouilli(p), where p randomly
        chosen from Uniform(0,1) at initialization
        """
        self.arm_count = arm_count
        self.generate_thetas()
        self.timestep = 0
        global stationary
        self.stationary = stationary
        self.actions = actions
        self.reward_table = rewards

    def generate_thetas(self):
        self.thetas = np.random.uniform(0, 1, self.arm_count)

    def get_reward_regret(self, arm):
        """ Returns random reward for arm action. Assumes actions are 0-indexed
        Args:
          arm is an int
        """
        self.timestep += 1
        if (self.stationary == False) and (self.timestep % 100 == 0):
            self.generate_thetas()
        # Simulate bernouilli sampling

        reward = self.reward_table[arm]
        regret = self.thetas.max() - self.thetas[arm]

        return reward, regret
