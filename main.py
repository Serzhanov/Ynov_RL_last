import gymnasium
import matplotlib.pyplot as plt
from IPython import display
import streamlit as st
from env import CustomEnv
# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env
from policy_ppo import ppo_policy_training, ppo_policy_testing
ENV = None
# create a small mobile environment for a single, centralized control agent
# pass rgb_array as render mode so the env can be rendered inside the notebook

# env = gymnasium.make("mobile-small-central-v0", render_mode="rgb_array")

# st.title("Mobile Environment Display")

# # Assuming you want to display the environment for a certain number of steps (e.g., 10)
# num_steps = 10
# env.reset()
# for step in range(num_steps):
#     # here, use random dummy actions by sampling from the action space
#     dummy_action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(dummy_action)

#     # render the environment

#     st.image(env.render(), use_column_width=True)
#     st.write(f"Step: {step+1}")

#     # Clear the previous plot for the next iteration
#     plt.clf()
#     display.clear_output(wait=True)


def init_env(ENV):
    ENV.reset()
    dummy_action = ENV.action_space.sample()
    obs, reward, terminated, truncated, info = ENV.step(dummy_action)
    return display_info(info)


def display_info(info):
    table_data = [(key, value) for key, value in info.items()]
    st.table(table_data)


def plot_env(ENV):
    st.image(ENV.render())


def main():
    global ENV
    # Streamlit app header
    st.sidebar.title("Streamlit App with Options")

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
        'ANN', 'Thompson', 'PPO', 'EXP3', 'UCB', 'DGPB'])

    if box_option == 'PPO':
        model = ppo_policy_training(ENV)
        info = ppo_policy_testing(model, ENV)
        display_info(info)
        plot_env(ENV)


if __name__ == '__main__':
    main()
