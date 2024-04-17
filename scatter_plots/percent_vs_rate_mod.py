import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_group_data(df, group_name, alpha=0.75, marker='o', x_offset=1.0, y_offset=0.5):
    """
    Plot data for a specified group with distinct colors.
    
    Parameters:
    df (DataFrame): Data source, filtered by group name.
    group_name (str): Group name to plot.
    alpha (float): Transparency of markers.
    marker (str): Marker type.
    x_offset (float): Horizontal offset for labels.
    y_offset (float): Vertical offset for labels.
    """
    group_data = df[df['group'] == group_name].reset_index(drop=True)  # Resetting index
    x = group_data['x']
    y = group_data['y']
    labels = group_data['label']
    sizes = group_data['size']

    # Use a consistent color for the entire group, mapped by unique group names
    unique_groups = df['group'].unique().tolist()
    color_index = unique_groups.index(group_name)
    color = plt.cm.tab10(color_index % 10)  # Using 'tab10' colormap for distinct colors

    # Plot data points
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=sizes[i], color=color, alpha=alpha, marker=marker, edgecolors='black', label=f'{group_name}' if i == 0 else "")
        plt.text(x[i] + x_offset, y[i] + y_offset, labels[i], ha='left', va='center', fontsize=9, color='black')

# Load data from a CSV file
df = pd.read_csv('scatter_plots/percent_vs_rate_data.csv')

# Create the plot
plt.figure(figsize=(10, 6))
for group in df['group'].unique():
    plot_group_data(df, group)

# Set plot limits and add grid
plt.xlim(0)
plt.ylim(0)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.xlabel('%')
plt.ylabel('%/day')
plt.title('Enhanced Scatter Plot with Dynamic Groups')
plt.legend(title='Group Legend')
plt.show()
