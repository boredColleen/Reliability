import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy

# Set parameters for Gaussian distributions
area_mean = 4.0       # mean area in mm^2
area_std = 0.1        # standard deviation of area in mm^2
height_mean = 100.0   # mean height in um
height_std = 20.0     # standard deviation of height in um

# Generate initial samples
num_samples = 100000
areas = norm.rvs(loc=area_mean, scale=area_std, size=num_samples)
heights = norm.rvs(loc=height_mean, scale=height_std, size=num_samples)

# Ensure all heights and areas are within the specified range
valid_heights = (heights > 0) & (heights < 200)
while not np.all(valid_heights):
    heights[~valid_heights] = norm.rvs(loc=height_mean, scale=height_std, size=np.sum(~valid_heights))
    valid_heights = (heights > 0) & (heights < 200)

valid_areas = (areas > 0) & (areas < 10)
while not np.all(valid_areas):
    areas[~valid_areas] = norm.rvs(loc=area_mean, scale=area_std, size=np.sum(~valid_areas))
    valid_areas = (areas > 0) & (areas < 10)

# Compute the water ingress (proportional to area and inversely proportional to height)
water_ingress = areas / heights

# Filter out extremely high water ingress values to improve visualization
max_water_ingress = 0.1  # Set a reasonable cap
filtered_water_ingress = water_ingress[water_ingress < max_water_ingress]

# Plot the histogram
plt.figure(figsize=(10, 6))
hist_data = plt.hist(filtered_water_ingress, bins=100, color='blue', alpha=0.7, density=True, label='Histogram')

# Fit a Cauchy distribution to the filtered data and plot it
loc, scale = cauchy.fit(filtered_water_ingress)
x_cauchy = np.linspace(min(filtered_water_ingress), max(filtered_water_ingress), 1000)
plt.plot(x_cauchy, cauchy.pdf(x_cauchy, loc, scale), 'r-', lw=2, label=f'Cauchy Fit: loc={loc:.2f}, scale={scale:.2f}')

# Fit a normal distribution to the filtered data and plot it
mean, std = norm.fit(filtered_water_ingress)
x_norm = np.linspace(min(filtered_water_ingress), max(filtered_water_ingress), 1000)
plt.plot(x_norm, norm.pdf(x_norm, mean, std), 'g-', lw=2, label=f'Normal Fit: mean={mean:.2f}, std={std:.2f}')

plt.title('Distribution of Water Ingress as a Function of Area and Height')
plt.xlabel('Water Ingress (arbitrary units)')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.show()
