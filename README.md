# Overview
This project centers around the exploration and implementation of coordination algorithms in wireless mobile networks, utilizing the `mobile-env` environment. The environment, characterized by its openness and minimalism, is tailored for the training and evaluation of coordination algorithms. Our approach involves the application of various machine learning algorithms, namely PPO, ANN, Thompson, UCB, EpsilonGreedy, and EXP3, to compare and observe their respective outcomes.

## Data Used
The project utilizes the `mobile-env` environment, which supports multi-agent and centralized reinforcement learning policies. It allows modeling user movement, connectivity to base stations, and offers flexibility in defining rewards, observations, and other aspects.

## Features
- Mobile environment.
- Different Reinforcement algorithms to apply on the environment with a user-friendly web interface.
- Algorithms generate the best possible action on the environment to get the best utility according to them.

## Reward and Action Configuration
- Actions are generated uniformly at random using the bounds of the environment.
- Rewards are determined based on the mean utility of the corresponding actions.

## Requirements
- Python 3.x
- Dependencies listed in `requirements.txt`

### Dependencies
Before making changes, ensure that you have the necessary dependencies installed, including Streamlit, Plotly Express, Pandas, and Folium. You can install them using `pip`:

```bash
python -m pip install -r requirements.txt
```

## User Guide

### Setting Up the Environment
1. Install the required dependencies.
2. Clone the `mobile-env` repository.
3. Open a web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).
4. Configure/Choose the `mobile-env` environment with relevant parameters.
5. Choose the coordination algorithm to evaluate (ANN, Policy Iteration, Thompson/UCB/Epsilon Greedy, EXP3, UCB).
6. Observe actions that have been chosen by algorithms and applied to the environment.

## Developer Guide
This section provides an overview of the code's architecture and guidelines for making modifications or extensions.

### Code Structure
The code is organized into several modules, classes, and functions to maintain a clear and structured design. Here's an overview of the main components:

- `main.py`: Entry point for the Streamlit application.
- `display.py`: Contains functions for creating plots of algo results.
- `env.py`: Contains environment architecture and configuration.
- `algorithm_application.py`: Contains functions for algorithm application and action generation.
- `ANN`:
  - `ann.py`: Contains Neural network architecture.
  - `nn_data_gen.py`: Contains data generation for neural network training/testing data.
- `PPO`:
  - `ppo_polcy.py`: Contains PPO architecture and its application.
- `RL_classes`:
  - `Bandit.py`: Defines the Bandit class representing a bandit problem for reinforcement learning.
    - `__init__(self, arm_count, actions, rewards=None)`: Initializes a bandit problem with a specified number of arms, possible actions, and optional pre-defined rewards.
  - `BetaAlgo.py`: Defines the BernThompson class for implementing the Bernoulli Thompson Sampling algorithm.
  - `EpsilonGreedy.py`: Defines the EpsilonGreedy class for implementing the Epsilon-Greedy algorithm.
  - `EXP3.py`: Defines the EXP3 class for implementing the EXP3 algorithm.
  - `UCB.py`: Defines the UCB class for implementing the Upper Confidence Bound (UCB) algorithm.

## Conclusions
The main conclusions drawn from the study are as follows:
1. The implemented coordination algorithms (ANN, Policy Iteration, Thompson, EXP3, UCB) showcase varying performances in the `mobile-env` environment.
2. Multi-cell selection in coordinated multipoint scenarios, particularly maximizing Quality of Experience (QoE) globally, poses challenges due to conflicting goals between individual user equipments (UEs) and base stations (BSs).