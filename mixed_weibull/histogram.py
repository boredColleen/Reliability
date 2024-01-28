import matplotlib.pyplot as plt
import numpy as np

# Provided data
times = np.array([80, 100, 430] + [504] * 20)
events = np.array([1, 1, 0] + [0] * 20)

# Filter out the times for observed events (failures)
failure_times = times[events == 1]

# Define time points (in hours) starting from 0, as multiples of 24 up to 504 hours
time_points = np.arange(0, 504 + 24, 24)

# Calculate failures at each time point using np.histogram
failures_at_time_points, _ = np.histogram(failure_times, bins=time_points)

# Calculate cumulative failures
cumulative_failures = np.cumsum(failures_at_time_points)

# Total number of samples (including censored data)
total_samples = len(times)

# Calculate failure probability at each time point
failure_probability = cumulative_failures / total_samples

# Width of the bars - slightly less than the interval for gaps
bar_width = (time_points[1] - time_points[0]) * 0.8

# Create the plot
plt.figure(figsize=(10, 6))

# Calculate and plot the surviving samples as an outline extending up to 1
surviving_probability = 1 - failure_probability
plt.bar(time_points[:-1] - bar_width / 2, surviving_probability, bottom=failure_probability, color='lightgreen', edgecolor='green', linewidth=1.5, width=bar_width, label='Surviving')

# Plot the failure probability with an outline
plt.bar(time_points[:-1] - bar_width / 2, failure_probability, color='red', edgecolor='black', linewidth=1.5, width=bar_width, label='Failed')

# Add labels and title
plt.xlabel('Time (h)')
plt.ylabel('Probability')
plt.title('Failure and Survival Probability over Time')
plt.xticks(time_points)  # Set x-ticks to match time_points

# Set x-axis limits to fit the bars with gaps, adding some space on both sides
#plt.xlim(time_points[0] - 1.5 * bar_width, time_points[-2] + 1.5 * bar_width)
plt.xlim(-bar_width, max(time_points) + bar_width/2)  # Set x-axis limits to fit the bars with gaps

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
