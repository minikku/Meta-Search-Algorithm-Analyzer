import os
import random

from matplotlib import pyplot as plt
import numpy as np


def per_function_plot(algo, dim, actual_vals, predict_vals, error_vals, func_no, ml_model_options):

    # Plot predicted vs. actual values
    markers = ['o', 's', '^', 'v', 'p', 'P', '*', 'X', 'D']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']
    legend_list = ['Actual']
    for model in ml_model_options:
        legend_list.append(str(model).split('(')[0])

    plt.figure(figsize=(10, 6))
    plt.plot(actual_vals[0], label='Actual', color='k', linestyle='solid', linewidth=3)
    for i, pred in enumerate(predict_vals):
        # plt.plot(actual_vals[i], pred, marker=markers[i], label=str(ml_model_options[i]).split('(')[0])
        plt.plot(predict_vals[i], label=str(ml_model_options[i]).split('(')[0], color=colors[i], linestyle='--')

    plt.xlabel('Iterations')
    plt.ylabel('Values')
    plt.title(f'{algo} {dim}D F{func_no}: actual and predicted values')
    plt.yscale('log')
    plt.legend(legend_list)
    plt.grid(True, axis='y', alpha=0.3)  # Add grid lines for y-axis
    plt.show()

    # Plot stacked bar chart for errors
    original_error_val = error_vals

    # y-axis auto adjustment
    tmp_error_vals = original_error_val[:]
    tmp_error_vals.sort(reverse=True)
    significant_gap = 300000
    error_val_threshold = tmp_error_vals[0]

    if len(tmp_error_vals) >= 2:
        first_vs_second = tmp_error_vals[0] - tmp_error_vals[1]
        if first_vs_second >= significant_gap:
            error_val_threshold = tmp_error_vals[1] * 1.3

    # Cutoff significant large values
    filtered_error_vals = [min(value, error_val_threshold) for value in original_error_val]

    legend_list = []
    for model in ml_model_options:
        legend_list.append(str(model).split('(')[0])

    plt.figure(figsize=(15, 20))
    bar_heights = plt.bar(np.arange(len(filtered_error_vals)), filtered_error_vals, color=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'],
            label=[str(model).split('(')[0] for model in ml_model_options])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(np.arange(len(filtered_error_vals)), [str(model).split('(')[0] for model in ml_model_options], rotation=45, ha='right')
    plt.title(f'{algo} {dim}D F{func_no}: RMSE')
    plt.legend(legend_list[1:])
    plt.grid(True, alpha=0.3)
    # plt.gca().get_yaxis().set_visible(False)

    # Set the y-axis limits
    plt.ylim(0, error_val_threshold+error_val_threshold*0.05)  # Set the maximum value for the y-axis

    # Add error values above bars
    for i, bar in enumerate(bar_heights):
        bar_height = bar.get_height()
        plt.annotate(
            # f'{bar_height:.2f}',
            f'{original_error_val[i]:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar_height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

    plt.show()


def aggregated_plot(fids, agg_df, ml_model_options):

    original_error_val = agg_df

    # y-axis auto adjustment
    tmp_error_vals = original_error_val[:]
    tmp_error_vals.sort(reverse=True)
    significant_gap = 300000
    error_val_threshold = tmp_error_vals[0]

    if len(tmp_error_vals) >= 2:
        first_vs_second = tmp_error_vals[0] - tmp_error_vals[1]
        if first_vs_second >= significant_gap:
            error_val_threshold = tmp_error_vals[1] * 1.3

    # Cutoff significant large values
    filtered_error_vals = [min(value, error_val_threshold) for value in original_error_val]

    legend_list = []
    for model in ml_model_options:
        legend_list.append(str(model).split('(')[0])

    plt.figure(figsize=(15, 20))
    bar_heights = plt.bar(np.arange(len(filtered_error_vals)), filtered_error_vals, color=['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'],
            label=[str(model).split('(')[0] for model in ml_model_options])
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.xticks(np.arange(len(filtered_error_vals)), [str(model).split('(')[0] for model in ml_model_options], rotation=45, ha='right')
    plt.title(
        f'{len(fids)} functions aggregated view: Accumulated RMSE over dimensions/functions for all compared models')
    plt.legend(legend_list[1:])
    plt.grid(True, alpha=0.3)
    plt.gca().get_yaxis().set_visible(False)

    # Set the y-axis limits
    plt.ylim(0, error_val_threshold+error_val_threshold*0.05)  # Set the maximum value for the y-axis

    # Add error values above bars
    for i, bar in enumerate(bar_heights):
        bar_height = bar.get_height()
        plt.annotate(
            # f'{bar_height:.2f}',
            f'{original_error_val[i]:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, bar_height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

    plt.show()
