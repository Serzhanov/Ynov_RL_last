import numpy as np
import random


def create_data(shape, values_range):
    data = [[random.randint(values_range[0], values_range[1])
             for j in range(shape[0])] for i in range(shape[1])]

    return data


def detect_utility(line1, line2, env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(line1)
    obs2, reward2, terminated2, truncated2, info2 = env.step(line2)
    if info['mean utility'] > info2['mean utility']:
        return 0
    return 1


def divide_data(data1, data2, env):
    train_data = []
    test_data = []
    for i in range(len(data1)):
        resp = detect_utility(data1[i], data2[i], env)
        if resp == 0:
            train_data.append(data1[i])
        else:
            test_data.append(data2[i])
    return train_data, test_data


def generate_data(env):
    action = env.action_space.sample()
    shape = (action.shape[0], 100)
    val_range = (0, np.max(action[0]))
    data1 = create_data(shape=shape, values_range=val_range)
    data2 = create_data(shape=shape, values_range=val_range)
    train_data, test_data = divide_data(data1, data2, env)
    return train_data, test_data
