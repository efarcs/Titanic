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

def plot_grouped_overlay_density(ax, sub_series:dict, boundary:int, n_points:int):

    if len(sub_series) < 2: 
            raise ValueError("Less than 2 values within sub_series, function is unusable.")
    
    plots = {}

    for label, series in sub_series.items():
        if len(series) < 2: 
            raise ValueError(f'{label} has less than 2 values, cannot fit to KDE.')
        mirrored = np.concatenate([series, 2 * boundary - series])
        plots[label] = stats.gaussian_kde(mirrored)

    

    

    stacked_values = np.concatenate(list(sub_series.values()))
    x = np.linspace(boundary,max(stacked_values), n_points)

    for label,kde in plots.items():
        y = kde(x) * 2
        ax.plot(x,y, label=label)











    


