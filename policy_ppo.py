from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import streamlit as st


def ppo_policy_training(env, total_timesteps=10):
    # train PPO agent on environment. this takes a while
    model = PPO(MlpPolicy, env, tensorboard_log='results_sb', verbose=1)
    # Create progress bar
    progress_bar = st.progress(0)

    # Update the progress bar during training
    for t in range(1, total_timesteps+1):
        model.learn(total_timesteps=1)

        # Update the progress bar value
        progress_bar.progress(t*10)
        t += 1

    # Remove the progress bar when training is complete
    # progress_bar.empty()
    st.success('Training Complete')
    return model


def ppo_policy_testing(model, env):
    # run one episode with the trained model
    obs, info = env.reset()
    progress_bar = st.progress(0)
    done = False
    i = 0
    while not done:
        action, _ = model.predict(obs)
        # perform step on simulation environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        i += 1
        progress_bar.progress(i)
    # progress_bar.progress(i)
    st.success('Test Complete')
    return info
