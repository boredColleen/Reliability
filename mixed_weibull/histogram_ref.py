import matplotlib.pyplot as plt
import numpy as np

# Total number of samples
total_samples = 24

# Define time points (in hours) starting from 0, as multiples of 24 up to 504 hours
time_points = np.arange(0, 504 + 24, 24)

# Number of new failures at each corresponding time point
# Adjust 'new_failures' according to your data
new_failures = np.array([0, 1, 2, 0, 4, 2, 1, 0, 1] + [0] * (len(time_points) - 9))

# Calculate cumulative failures
cumulative_failures = np.cumsum(new_failures)

# Calculate failure probability at each time point
failure_probability = cumulative_failures / total_samples

# Width of the bars - slightly less than the interval for gaps
bar_width = (time_points[1] - time_points[0]) * 0.8

# Create the plot
plt.figure(figsize=(8, 2))

# Plot the number of surviving samples as an outline extending up to 1
plt.bar(time_points - bar_width/2, 1 - failure_probability, bottom=failure_probability, color='lightgreen', edgecolor='green', linewidth=1.5, width=bar_width, label='Surviving')

# Plot the failure probability with an outline
plt.bar(time_points - bar_width/2, failure_probability, color='red', edgecolor='black', linewidth=1.5, width=bar_width, label='Failed')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Probability')
plt.title('Failure and Survival Probability over Time')
plt.xticks(time_points, time_points)  # Set x-ticks to match time_points (numbers only)
plt.xlim(-bar_width, max(time_points) + bar_width/2)  # Set x-axis limits to fit the bars with gaps

# Add a legend
plt.legend()

# Automatically adjust subplot params so that the subplot(s) fits in to the figure area
plt.tight_layout()

# Show the plot
plt.show()