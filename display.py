import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def multi_plot_data(data, names):
    fig, axes = plt.subplots(nrows=len(names), figsize=(8, 6))
    # Plot data on each subplot
    for i, ax in enumerate(axes):
        x = np.arange(data[i].size)
        ax.plot(x, data[i], 'o', markersize=2, label=names[i])
        ax.legend(loc='upper right', prop={'size': 8}, numpoints=1)
        ax.set_title(f'Subplot {i + 1}')
        ax.set_xlabel('X-axis Label (simulations)')
        ax.set_ylabel('Y-axis Label (regrets)')

    st.pyplot(fig)
