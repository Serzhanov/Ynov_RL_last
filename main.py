import gymnasium
import matplotlib.pyplot as plt
from IPython import display
import streamlit as st

# importing mobile_env automatically registers the predefined scenarios in Gym
import mobile_env

# create a small mobile environment for a single, centralized control agent
# pass rgb_array as render mode so the env can be rendered inside the notebook
env = gymnasium.make("mobile-small-central-v0", render_mode="rgb_array")

st.title("Mobile Environment Display")

# Assuming you want to display the environment for a certain number of steps (e.g., 10)
num_steps = 10
env.reset()
for step in range(num_steps):
    # here, use random dummy actions by sampling from the action space
    dummy_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(dummy_action)

    # render the environment

    st.image(env.render(), use_column_width=True)
    st.write(f"Step: {step+1}")

    # Clear the previous plot for the next iteration
    plt.clf()
    display.clear_output(wait=True)
