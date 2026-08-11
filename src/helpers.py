import pandas as pd
import numpy as np
import regex as re
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

#Function to help plot overlayed graphs, i.e comparing number of survivors vs total
def plot_grouped_overlay(ax , total_df, sub_df, categories, width = 0.25):

    x = np.arange(len(total_df))
    n = len(categories)
    for i, cat in enumerate(categories):
        offset = width * (i - (n-1) / 2)
        bars = ax.bar(x + offset, total_df[cat], width=width, color = "lightgray")
        ax.bar_label(bars, labels = [cat[0]] * len(bars))
        ax.bar(x + offset, sub_df[cat], width=width, color = "green")
    ax.set_xticks(x)
    ax.set_xticklabels(total_df.index)
