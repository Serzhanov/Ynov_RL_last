import gymnasium
import streamlit as st
from env import CustomEnv
from algorithm_application import generate_actions, generate_rewards
# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env
from PPO.ppo_policy import ppo_policy_training, ppo_policy_testing
from ANN.ann import generate_data
from ANN.nn_data_gen import init_ANN
import numpy as np
from algorithm_application import experiment
from display import multi_plot_data, plot_data
from RL_classes.EXP3 import EXP3

ENV = None


def display_and_plot(env, info):
    display_info(info)
    plot_env(env)


def init_env(ENV):
    '''Initiate enviroment'''
    ENV.reset()
    dummy_action = ENV.action_space.sample()
    obs, reward, terminated, truncated, info = ENV.step(dummy_action)
    return display_info(info)


def display_info(info):
    '''Display info of actions'''
    table_data = [(key, value) for key, value in info.items()]
    st.table(table_data)


def plot_env(ENV):
    '''Plot current enviroment state'''
    st.image(ENV.render())


def main():
    global ENV
    # Streamlit app header
    st.sidebar.title("Create or Choose your enviroment")

    # Add options using radio buttons
    option = st.sidebar.radio("Choose a enviroment size:", [
        "Small", "Medium", "Large", "Custom"])

    # Display content based on the selected option
    if option == "Small":
        ENV = gymnasium.make("mobile-small-central-v0",
                             render_mode="rgb_array")
        st.write(f"You chosed SMALL Enviroment")
    elif option == "Medium":
        ENV = gymnasium.make("mobile-medium-central-v0",
                             render_mode="rgb_array")
        st.write(f"You chosed MEDIUM Enviroment")
    elif option == "Large":
        ENV = gymnasium.make("mobile-large-central-v0",
                             render_mode="rgb_array")
        st.write(f"You chosed LARGE Enviroment")
    elif option == "Custom":
        st.sidebar.write("Custom")

        user_num = st.sidebar.number_input(
            "Enter a number of users : ", value=3)
        station_num = st.sidebar.number_input(
            "Enter a number of stations : ", value=3)
        st.write(f"You entered: {user_num} and {station_num}")
        ENV = CustomEnv(users_num=int(user_num), stations_num=int(
            station_num), render_mode="rgb_array")

    init_env(ENV)
    plot_env(ENV)

    st.title('Policy')
    box_option = st.selectbox("Select an option:", [
        'None', 'ANN', 'Thompson/UCB/EpsilonGreedy', 'PPO', 'EXP3', 'DGPB'])

    if box_option == 'PPO':
        model = ppo_policy_training(ENV)
        info = ppo_policy_testing(model, ENV)
        display_and_plot(ENV)

    if box_option == 'ANN':

        train_data, test_data = generate_data(ENV)
        if len(train_data) == 0:
            st.write('RERUN by using R key')
            return
        size = len(train_data[0])

        st.write(
            f"Your input and output for this enviroment will be {size} it depends on env configuration")
        hidden_layers_num = st.number_input(
            "How many hidden layers do you want :", value=2, max_value=10, min_value=2)
        hidden_layers_size = st.number_input(
            "Their input/output size (for simplicity they would be the same (more work is coming)):", value=3, max_value=10, min_value=3)
        epochs = st.number_input("Number of epochs: ",
                                 value=5, min_value=1, max_value=50)
        learning_rate = st.number_input(
            "Learning rate: ", value=0.01, min_value=0.001, max_value=0.1)
        if type(learning_rate) == 'str':
            return
        nn = init_ANN(np.matrix(train_data), np.matrix(test_data), epochs, hidden_layers_num,
                      size=size, hidden_layers_size=hidden_layers_size, learning_rate=learning_rate)
        dummy_action = ENV.action_space.sample()
        nn_action = nn.forward(dummy_action)  # might return float
        st.write('Testing')
        nn_action = [int(x) for x in nn_action[0]]
        obs, reward, terminated, truncated, info = ENV.step(nn_action)
        display_and_plot(ENV, info)

    if box_option == 'Thompson/UCB/EpsilonGreedy':
        actions = generate_actions(ENV)
        reward_table = generate_rewards(ENV, actions)
        timesteps = st.number_input(
            "Timesteps :", value=5, max_value=1000, min_value=5)
        simulations = st.number_input(
            'Simulations "', value=5, max_value=1000, min_value=5)
        regrets, names, best_actions_by_algo = experiment(actions=actions, rewards=reward_table, arm_count=len(actions),
                                                          simulations=simulations, timesteps=timesteps)
        multi_plot_data(regrets, names=names)
        st.write('Best action performance')
        algo_choice = st.selectbox("Select best algo for action:", [
            'Thompson', 'UCB', 'Epsilon Greedy'])
        if algo_choice == 'Thompson':
            action = best_actions_by_algo['thompson']
        if algo_choice == 'UCB':
            action = best_actions_by_algo['ucb']
        if algo_choice == 'Epsilon Greedy':
            action = best_actions_by_algo['epsilon-greedy']
        action = [int(x) for x in action]
        obs, reward, terminated, truncated, info = ENV.step(action)
        display_and_plot(ENV, info)

    if box_option == 'EXP3':
        actions = generate_actions(ENV)
        reward_table = generate_rewards(ENV, actions)
        simulations = st.number_input(
            "Simulations :", value=5, max_value=1000, min_value=5)
        gamma = st.number_input(
            "Gamma :", value=0.1, max_value=1.0, min_value=0.1, format="%.2f", step=1.)
        eta = st.number_input(
            "Eta :", value=0.1, max_value=1.0, min_value=0.1, format="%.2f", step=1.)
        exp3_algo = EXP3(num_arms=len(actions), simulations_num=simulations,
                         rewards=reward_table, actions=actions, gamma=gamma, eta=eta)
        simulations, rewards = exp3_algo.simulate()
        plot_data(simulations, rewards, xlabel='simulations', ylabel='rewards')
        st.write('Best action performance')
        action = exp3_algo.get_best_action()
        action = [int(x) for x in action]
        obs, reward, terminated, truncated, info = ENV.step(action)
        display_and_plot(ENV, info)


if __name__ == '__main__':
    main()
