import numpy as np
import scipy.stats as stats
from scipy.optimize import differential_evolution

def simulate_corrosion(std_dev_pinhole_effect, time_factor):
    base_wvtr = 170
    mean_pinhole_effect = 30
    mean_electrolyte_effect = 0.98
    std_dev_electrolyte_effect = 0.24  # Set as a constant

    pinhole_effects = np.random.normal(mean_pinhole_effect, std_dev_pinhole_effect, size=10000)
    electrolyte_effects = np.random.normal(mean_electrolyte_effect, std_dev_electrolyte_effect, size=10000)

    final_wvtr = base_wvtr + pinhole_effects
    valid_indices = final_wvtr > 0
    valid_wvtr = final_wvtr[valid_indices]
    valid_electrolyte_effects = electrolyte_effects[valid_indices]

    corrosion_time = time_factor * 24 * valid_electrolyte_effects / valid_wvtr
    valid_corrosion_time = corrosion_time[(corrosion_time > 0) & (corrosion_time <= 500)]

    shape_ct, loc_ct, scale_ct = stats.weibull_min.fit(valid_corrosion_time, floc=0)
    return shape_ct, scale_ct

def objective(x):
    std_dev_pinhole_effect, time_factor = x
    shape, scale = simulate_corrosion(std_dev_pinhole_effect, time_factor)
    target_shape = 1.23
    target_scale = 77.7
    return (shape - target_shape) ** 2 + (scale - target_scale) ** 2  # Ensure this is a single scalar value

def callback_fn(xk, convergence):
    shape, scale = simulate_corrosion(xk[0], xk[1])
    print(f"Std Dev Pinhole: {xk[0]:.2f}, Time Factor: {xk[1]:.2f}, Shape: {shape:.2f}, Scale: {scale:.2f}, Convergence: {convergence:.6f}")

def optimize_std_devs():
    bounds = [(0, 200), (100, 600)]
    result = differential_evolution(objective, bounds, seed=42, updating='deferred', callback=callback_fn, maxiter=100, tol=0.01)
    if result.success:
        return result.x
    else:
        print("Optimization failed: Maximum number of iterations has been exceeded.")
        return result.x

optimized_std_devs = optimize_std_devs()
shape, scale = simulate_corrosion(optimized_std_devs[0], optimized_std_devs[1])
print("Optimized Std Dev for Pinhole Effects: {:.2f}".format(optimized_std_devs[0]))
print("Optimized Time Factor: {:.2f}".format(optimized_std_devs[1]))
print("Resulting Shape: {:.2f}".format(shape))
print("Resulting Scale: {:.2f}".format(scale))
