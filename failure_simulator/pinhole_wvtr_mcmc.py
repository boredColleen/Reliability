import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as stats

# Generate synthetic data
np.random.seed(42)
n_samples = 10000

# Constants
base_wvtr = 170  # Base WVTR without pinholes

# Distribution parameters for direct WVTR effect of pinholes
mean_pinhole_effect = 30  # Mean increase in WVTR due to pinholes
a = 146.04
std_dev_pinhole_effect = a  # Standard deviation of the increase

# Distribution parameters for electrolyte concentration effects on WVTR
mean_electrolyte_effect = 1.00
std_dev_electrolyte_effect = 0.31

# Generate pinhole effects on WVTR
pinhole_effects = np.random.normal(mean_pinhole_effect, std_dev_pinhole_effect, size=n_samples)

# Generate initial electrolyte effects
initial_electrolyte_effects = np.random.normal(mean_electrolyte_effect, std_dev_electrolyte_effect, size=n_samples)

# Calculate final WVTR by combining base WVTR, pinhole effects, and initial electrolyte effects
final_wvtr = base_wvtr + pinhole_effects

# Ensure all WVTR values are positive for realistic modeling
valid_indices = final_wvtr > 0
valid_wvtr = final_wvtr[valid_indices]

# Filter electrolyte effects to match the valid indices of WVTR
valid_electrolyte_effects = initial_electrolyte_effects[valid_indices]

# Now use valid_electrolyte_effects in your corrosion time calculation
b = 445.12
corrosion_time = b * 24 * (valid_electrolyte_effects * 0.85) / valid_wvtr

# Filter out non-physical and excessive corrosion times
valid_corrosion_time = corrosion_time[(corrosion_time > 0) & (corrosion_time <= 500)]

# Bayesian model
with pm.Model() as model:
    # Priors
    pinhole_effect = pm.Normal('pinhole_effect', mu=mean_pinhole_effect, sigma=std_dev_pinhole_effect)
    electrolyte_effect = pm.Normal('electrolyte_effect', mu=mean_electrolyte_effect, sigma=std_dev_electrolyte_effect)
    
    # Likelihood
    wvtr = pm.Normal('wvtr', mu=base_wvtr + pinhole_effect, sigma=std_dev_pinhole_effect, observed=valid_wvtr)
    corrosion = pm.Normal('corrosion', mu=b * 24 * (electrolyte_effect * 0.85) / wvtr, sigma=std_dev_electrolyte_effect, observed=valid_corrosion_time)
    
    # Posterior
    trace = pm.sample(1000, tune=2000, return_inferencedata=True)

# Plotting the results
az.plot_trace(trace)
plt.show()

# Summary of the trace
summary = az.summary(trace)
print(summary)

# Plotting both distributions and their Weibull fits
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plotting WVTR distribution and Weibull fit
shape_wvtr, loc_wvtr, scale_wvtr = stats.weibull_min.fit(valid_wvtr, floc=0)
axs[0].hist(valid_wvtr, bins=50, alpha=0.6, color='green', density=True)
x_wvtr = np.linspace(valid_wvtr.min(), valid_wvtr.max(), 100)
axs[0].plot(x_wvtr, stats.weibull_min.pdf(x_wvtr, shape_wvtr, loc_wvtr, scale_wvtr), 'r-')
axs[0].set_xlabel('WVTR')
axs[0].set_ylabel('Density')
axs[0].legend()

# Plotting Corrosion Time distribution and Weibull fit
shape_ct, loc_ct, scale_ct = stats.weibull_min.fit(valid_corrosion_time, floc=0)
axs[1].hist(valid_corrosion_time, bins=50, alpha=0.6, color='blue', density=True, label='Corrosion Time Data')
x_ct = np.linspace(valid_corrosion_time.min(), valid_corrosion_time.max(), 100)
axs[1].plot(x_ct, stats.weibull_min.pdf(x_ct, shape_ct, loc_ct, scale_ct), 'r-', label=f'shape={shape_ct:.2f}, scale={scale_ct:.2f}')
axs[1].set_xlabel('Corrosion Time (h)')
axs[1].legend()

plt.show()

# Print Weibull parameters for reference
print(f"WVTR Weibull Parameters: shape={shape_wvtr:.2f}, scale={scale_wvtr:.2f}")
print(f"Corrosion Time Weibull Parameters: shape={shape_ct:.2f}, scale={scale_ct:.2f}")
