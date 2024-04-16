import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_group_data(df, group_number, alpha=0.75, marker='o', x_offset=1.0, y_offset=0.5):
    """
    Plot data for a specified group with distinct colors.
    
    Parameters:
    df (DataFrame): Data source.
    group_number (int): Group number to plot.
    alpha (float): Transparency of markers.
    marker (str): Marker type.
    x_offset (float): Horizontal offset for labels.
    y_offset (float): Vertical offset for labels.
    """
    # Construct column names based on group number
    x_col = f'x_group{group_number}'
    y_col = f'y_group{group_number}'
    label_col = f'label_group{group_number}'
    size_col = f'size_group{group_number}'

    # Extract data
    x = df[x_col]
    y = df[y_col]
    labels = df[label_col]
    sizes = df[size_col]

    # Use a predefined color map for distinct colors
    colormap = plt.get_cmap('tab10')  # 'tab10' supports up to 10 distinct colors
    color = colormap(group_number % 10)  # Cycle through colors if more than 10 groups

    # Plot data points
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=sizes[i], color=color, alpha=alpha, marker=marker, edgecolors='black', label=f'Group {group_number}' if i == 0 else "")
        plt.text(x[i] + x_offset, y[i] + y_offset, labels[i], ha='left', va='center', fontsize=9, color='black')

# Load data from a CSV file
df = pd.read_csv('scatter_plots/percent_vs_rate_data.csv') 

# Create the plot
plt.figure(figsize=(10, 6))
plot_group_data(df, 1)
plot_group_data(df, 2)

# Set plot limits and add grid
plt.xlim(0, 100)
plt.ylim(0)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.xlabel('%')
plt.ylabel('%/day')
plt.title('Enhanced Scatter Plot with Contrasting Group Colors')
plt.legend(title='Group Legend')
plt.show()
