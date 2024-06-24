import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def validate_and_load_csv(file_path):
    """
    Validate and load a CSV file if valid.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame | None: Loaded DataFrame if the file is valid, None otherwise.
    """
    required_columns = {'group', 'label', 'x', 'y', 'size'}
    try:
        df = pd.read_csv(file_path)
        if not set(required_columns).issubset(df.columns):
            print("Error: CSV file is missing one or more required columns.")
            return None
        if df[list(required_columns)].isnull().any().any():
            print("Error: CSV file contains missing values in required columns.")
            return None
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in ['x', 'y', 'size']):
            print("Error: Some columns contain non-numeric data.")
            return None
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_contours(df, margin=0.1):
    """
    Plot density contours for the entire dataset, considering the 'size' as weights, and within an extended range.
    
    Parameters:
    df (DataFrame): Data source for plotting contours.
    margin (float): Percentage to extend the plot boundaries.
    """
    x = df['x']
    y = df['y']
    sizes = df['size']
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=sizes, bw_method='silverman')
    
    # Extend the x and y ranges by the specified margin
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = (x_max - x_min) * margin
    y_range = (y_max - y_min) * margin
    x_grid = np.linspace(x_min - x_range, x_max + x_range, 200)
    y_grid = np.linspace(y_min - y_range, y_max + y_range, 200)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    z = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)
    plt.contour(x_mesh, y_mesh, z, levels=15, colors='black', linewidths=0.5, alpha=0.8)

def plot_group_data(df, group_name):
    """
    Plot data for a specified group with distinct colors.
    
    Parameters:
    df (DataFrame): Data source, filtered by group name.
    group_name (str): Group name to plot.
    """
    group_data = df[df['group'] == group_name]
    plt.scatter(group_data['x'], group_data['y'], s=group_data['size'] * 1, alpha=0.7, label=group_name)

file_path = 'scatter_plots/percent_vs_rate_data.csv'
df = validate_and_load_csv(file_path)
if df is None:
    raise ValueError("The CSV file is invalid or not formatted correctly. Please check the file and try again.")

# Create the plot
plt.figure(figsize=(6, 4))
plot_contours(df)
for group in df['group'].unique():
    plot_group_data(df, group)

# Expand plot limits as before, but now contours are also expanded
x_range = (df['x'].max() - df['x'].min()) * 0.1
y_range = (df['y'].max() - df['y'].min()) * 0.1
plt.xlim(df['x'].min() - x_range, df['x'].max() + x_range)
plt.ylim(df['y'].min() - y_range, df['y'].max() + y_range)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.xlabel('%')
plt.ylabel('%/day')
plt.title('Enhanced Scatter Plot with Weighted Contour Overlays')
plt.legend(title='Group Legend')
plt.show()
