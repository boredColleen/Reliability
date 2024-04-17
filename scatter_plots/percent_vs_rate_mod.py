import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import pandas as pd

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
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert required_columns set to a list for pandas indexing
        required_columns_list = list(required_columns)
        
        # Check for required columns
        if not set(required_columns_list).issubset(df.columns):
            print("Error: CSV file is missing one or more required columns.")
            return None
        
        # Check for any NaN values in the required columns
        if df[required_columns_list].isnull().any().any():
            print("Error: CSV file contains missing values in required columns.")
            return None
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['x']) or not pd.api.types.is_numeric_dtype(df['y']):
            print("Error: 'x' or 'y' columns contain non-numeric data.")
            return None
        
        if not pd.api.types.is_numeric_dtype(df['size']):
            print("Error: 'size' column contains non-numeric data.")
            return None
        
        print("CSV file is valid.")
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_group_data(df, group_name, alpha=0.5, marker='o', x_offset=1.0, y_offset=0.5):
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
    labels = group_data['label']
    x = group_data['x']
    y = group_data['y']
    sizes = group_data['size']

    # Use a consistent color for the entire group, mapped by unique group names
    unique_groups = df['group'].unique().tolist()
    color_index = unique_groups.index(group_name)
    color = plt.cm.tab10(color_index % 10)  # Using 'tab10' colormap for distinct colors

    # Plot data points
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=sizes[i], color=color, alpha=alpha, marker=marker, edgecolors='black', label=f'{group_name}' if i == 0 else "")
        plt.text(x[i] + x_offset, y[i] + y_offset, labels[i], ha='left', va='center', fontsize=9, color='black')


file_path = 'scatter_plots/percent_vs_rate_data.csv'
df = validate_and_load_csv(file_path)
if df is not None:
    print("CSV file is valid and has been loaded successfully. Proceed with data processing.")
    # Additional data processing can be added here, e.g., plotting.
else:
    raise ValueError("The CSV file is invalid or not formatted correctly. Please check the file and try again.")


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
