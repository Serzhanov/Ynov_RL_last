import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def multi_plot_data(data, names, xlabel='simulations', ylabel='regrets'):
    fig, axes = plt.subplots(nrows=len(names), figsize=(8, 6))
    # Plot data on each subplot
    for i, ax in enumerate(axes):
        x = np.arange(data[i].size)
        ax.plot(x, data[i], 'o', markersize=2, label=names[i])
        ax.legend(loc='upper right', prop={'size': 8}, numpoints=1)
        ax.set_title(f'Subplot {i + 1}')
        ax.set_xlabel(f'X-axis Label ({xlabel})')
        ax.set_ylabel(f'Y-axis Label ({ylabel})')

    st.pyplot(fig)


def plot_data(dataX, dataY, xlabel='simulations', ylabel='regrets', name='EXP3'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dataX, dataY, 'o', markersize=2, label=name)
    ax.legend(loc='upper right', prop={'size': 8}, numpoints=1)
    ax.set_title('Single Subplot')
    ax.set_xlabel(f'X-axis Label ({xlabel})')
    ax.set_ylabel(f'Y-axis Label ({ylabel})')

    st.pyplot(fig)
