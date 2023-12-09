import numpy as np
import random


def create_data(shape, values_range):
    """
    Generates random data in the form of a 2D list with a specified shape and value range.

    Parameters:
    - shape (tuple): Shape of the 2D list.
    - values_range (tuple): Range of random values (min, max).

    Returns:
    - list: 2D list containing random integers within the specified range.
    """
    data = [[random.randint(values_range[0], values_range[1])
             for j in range(shape[0])] for i in range(shape[1])]

    return data


def detect_utility(line1, line2, env):
    """
    Compares the mean utility of two lines of data in a simulated environment.

    Parameters:
    - line1, line2: Two lines of data.
    - env: Environment for simulation.

    Returns:
    - int: 0 if mean utility of line1 > line2, 1 otherwise.
    """
    env.reset()
    obs, reward, terminated, truncated, info = env.step(line1)
    obs2, reward2, terminated2, truncated2, info2 = env.step(line2)
    if info['mean utility'] > info2['mean utility']:
        return 0
    return 1


def divide_data(data1, data2, env):
    """
    Divides data into training and testing sets based on simulated utility comparison.

    Parameters:
    - data1, data2: Two sets of data.
    - env: Environment for simulation.

    Returns:
    - tuple: Training and testing sets.
    """
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
    """
    Generates two sets of random data and divides them into training and testing sets.

    Parameters:
    - env: Environment for simulation.

    Returns:
    - tuple: Training and testing sets.
    """
    action = env.action_space.sample()
    shape = (action.shape[0], 100)
    val_range = (0, np.max(action[0]))
    data1 = create_data(shape=shape, values_range=val_range)
    data2 = create_data(shape=shape, values_range=val_range)
    train_data, test_data = divide_data(data1, data2, env)
    return train_data, test_data
