import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Constants
base_wvtr = 10  # Base WVTR without pinholes

# Distribution parameters for direct WVTR effect of pinholes
mean_pinhole_effect = 9  # Mean increase in WVTR due to pinholes
std_dev_pinhole_effect = 0.5  # Standard deviation of the increase

# Distribution parameters for electrolyte concentration effects on WVTR
mean_electrolyte_effect = 1.0
std_dev_electrolyte_effect = 0.2

# Generate pinhole effects on WVTR
pinhole_effects = np.random.normal(mean_pinhole_effect, std_dev_pinhole_effect, size=10000)

# Generate electrolyte effects
electrolyte_effects = np.random.normal(mean_electrolyte_effect, std_dev_electrolyte_effect, size=10000)

# Calculate final WVTR by combining base WVTR, pinhole effects, and electrolyte effects
final_wvtr = (base_wvtr + pinhole_effects) * electrolyte_effects

# Fit a Weibull distribution to the final WVTR data
shape, loc, scale = stats.weibull_min.fit(final_wvtr, floc=0)

# Plot the histogram of the WVTR distribution and the Weibull fit
plt.figure(figsize=(10, 6))
plt.hist(final_wvtr, bins=50, alpha=0.6, color='g', density=True, label='WVTR Data')
x = np.linspace(final_wvtr.min(), final_wvtr.max(), 100)
plt.plot(x, stats.weibull_min.pdf(x, shape, loc, scale), 'r-', label=f'Weibull Fit: shape={shape:.2f}, scale={scale:.2f}')
plt.title('Distribution of WVTR with Weibull Fit')
plt.xlabel('WVTR')
plt.ylabel('Density')
plt.legend()
plt.show()
