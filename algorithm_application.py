
import numpy as np
import random
from RL_classes.Bandit import Bandit
from RL_classes.BetaAlgo import BernThompson
from RL_classes.EpsilonGreedy import EpsilonGreedy
from RL_classes.Ucb import UCB


def get_utility(env, line):
    '''Get utility rate of Network enviroment'''
    env.reset()
    obs, reward, terminated, truncated, info = env.step(line)
    return info['mean utility']


def generate_actions(env):
    '''Generate random actions'''
    action = env.action_space.sample()
    shape = (action.shape[0], 100)
    val_range = (0, np.max(action[0]))
    data = [[random.randint(val_range[0], val_range[1])
             for j in range(shape[0])] for i in range(shape[1])]
    return data


def generate_rewards(env, actions):
    '''Generate rewards by taken actions'''
    reward_table = []
    for i in range(len(actions)):
        reward_table.append(get_utility(env, actions[i]))
    return reward_table


def simulate(simulations, timesteps, arm_count, Algorithm, actions, rewards):
    """ Simulates the algorithm over 'simulations' epochs """
    sum_regrets = np.zeros(timesteps)
    best_actions = {}
    for e in range(simulations):
        bandit = Bandit(arm_count, actions, rewards=rewards)
        algo = Algorithm(bandit)
        regrets = np.zeros(timesteps)
        for i in range(timesteps):
            action = algo.get_action()
            reward, regret = algo.get_reward_regret(action)
            regrets[i] = regret
        sum_regrets += regrets
        best_actions[algo.name()] = algo.get_best_action()
    mean_regrets = sum_regrets / simulations
    return mean_regrets, best_actions


def experiment(arm_count, actions, rewards, timesteps=100, simulations=100):
    """ 
    Standard setup across all experiments 
    Args:
      timesteps: (int) how many steps for the algo to learn the bandit
      simulations: (int) number of epochs
    """
    algos = [EpsilonGreedy, UCB, BernThompson]
    regrets = []
    names = []
    best_actions_by_algo = {}
    for algo in algos:
        mean_regrets, best_actions = simulate(simulations, timesteps,
                                              arm_count, algo, actions, rewards)
        regrets.append(mean_regrets)
        names.append(algo.name())
        if algo.name() not in best_actions_by_algo.keys():
            best_actions_by_algo[algo.name()] = best_actions[algo.name()]
        elif best_actions[algo.name()] > best_actions_by_algo[algo.name()]:
            best_actions_by_algo[algo.name()] = best_actions[algo.name()]

    return regrets, names, best_actions_by_algo
