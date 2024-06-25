import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Constants
base_wvtr = 170  # Base WVTR without pinholes
mean_pinhole_effect = 20  # Mean increase in WVTR due to pinholes
a_actual = 70
std_dev_pinhole_effect = a_actual  # Standard deviation of the increase

mean_electrolyte_effect = 1.0
std_dev_electrolyte_effect = 1.0

# Generate pinhole effects on WVTR
pinhole_effects = np.random.normal(mean_pinhole_effect, std_dev_pinhole_effect, size=10000)

# Generate initial electrolyte effects
initial_electrolyte_effects = np.random.normal(mean_electrolyte_effect, std_dev_electrolyte_effect, size=10000)

# Calculate final WVTR by combining base WVTR, pinhole effects, and initial electrolyte effects
final_wvtr = base_wvtr + pinhole_effects

# Ensure all WVTR values are positive for realistic modeling
valid_indices = final_wvtr > 0
valid_wvtr = final_wvtr[valid_indices]

# Filter electrolyte effects to match the valid indices of WVTR
valid_electrolyte_effects = initial_electrolyte_effects[valid_indices]

# Now use valid_electrolyte_effects in your corrosion time calculation
b_actual = 375
corrosion_time = b_actual * 24 * valid_electrolyte_effects / valid_wvtr

# Filter out non-physical and excessive corrosion times
valid_corrosion_time = corrosion_time[(corrosion_time > 0) & (corrosion_time <= 500)]

# Bayesian inference with PyMC
with pm.Model() as model:
    # Priors for the unknown parameters
    a = pm.Normal('a', mu=70, sigma=5)
    b = pm.Normal('b', mu=375, sigma=10)

    # Likelihood for the observed corrosion times
    # Using the shape parameter and the observed corrosion times to estimate 'a' and 'b'
    y_obs = pm.Weibull('y_obs', alpha=1.23, beta=b * (24 * a) / valid_corrosion_time, observed=valid_corrosion_time)

    # Posterior distribution sampling
    trace = pm.sample(1000, cores=1)

# Analyze results
pm.plot_posterior(trace)
plt.show()

# Extract and print estimated values
a_est = np.mean(trace.posterior['a'].values)
b_est = np.mean(trace.posterior['b'].values)
print(f"Estimated a: {a_est:.2f}")
print(f"Estimated b: {b_est:.2f}")

# Weibull fit for valid WVTR data
shape_wvtr, loc_wvtr, scale_wvtr = stats.weibull_min.fit(valid_wvtr, floc=0)

# Weibull fit for valid corrosion time
shape_ct, loc_ct, scale_ct = stats.weibull_min.fit(valid_corrosion_time, floc=0)

# Plotting both distributions and their Weibull fits
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plotting WVTR distribution and Weibull fit
axs[0].hist(valid_wvtr, bins=50, alpha=0.6, color='green', density=True, label='WVTR Data')
x_wvtr = np.linspace(valid_wvtr.min(), valid_wvtr.max(), 100)
axs[0].plot(x_wvtr, stats.weibull_min.pdf(x_wvtr, shape_wvtr, loc_wvtr, scale_wvtr), 'r-', label=f'Weibull Fit: shape={shape_wvtr:.2f}, scale={scale_wvtr:.2f}')
axs[0].set_title('Distribution of Valid WVTR with Weibull Fit')
axs[0].set_xlabel('WVTR')
axs[0].set_ylabel('Density')
axs[0].legend()

# Plotting Corrosion Time distribution and Weibull fit
axs[1].hist(valid_corrosion_time, bins=50, alpha=0.6, color='blue', density=True, label='Corrosion Time Data')
x_ct = np.linspace(valid_corrosion_time.min(), valid_corrosion_time.max(), 100)
axs[1].plot(x_ct, stats.weibull_min.pdf(x_ct, shape_ct, loc_ct, scale_ct), 'r-', label=f'Weibull Fit: shape={shape_ct:.2f}, scale={scale_ct:.2f}')
axs[1].set_title('Distribution of Corrosion Time (Valid Range) with Weibull Fit')
axs[1].set_xlabel('Corrosion Time (hours)')
axs[1].legend()

plt.show()

# Print Weibull parameters for reference
print(f"WVTR Weibull Parameters: shape={shape_wvtr:.2f}, scale={scale_wvtr:.2f}")
print(f"Corrosion Time Weibull Parameters: shape={shape_ct:.2f}, scale={scale_ct:.2f}")
