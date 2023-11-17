import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Known data points
x_known = np.array([0, 1, 2, 3, 4])
y_known = np.array([0, 1, 4, 9, 16])

# Interpolation function (cubic)
interpolation_func = interp1d(x_known, y_known, kind='cubic')

# Extrapolation function (cubic)
extrapolation_func = interp1d(x_known, y_known, kind='cubic', fill_value='extrapolate')

# Points for interpolation and extrapolation
x_interpolation = np.linspace(0, 4, 100)  # Within the range of known points
x_extrapolation = np.linspace(-1, 5, 100)  # Outside the range of known points

# Calculate interpolated and extrapolated values
y_interpolation = interpolation_func(x_interpolation)
y_extrapolation = extrapolation_func(x_extrapolation)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Interpolation plot
ax1.scatter(x_known, y_known, color='black', label='Known Data Points')
ax1.plot(x_interpolation, y_interpolation, color='blue', label='Interpolation')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Interpolation')
ax1.grid(True)
ax1.legend()

# Extrapolation plot
ax2.scatter(x_known, y_known, color='black', label='Known Data Points')
ax2.plot(x_extrapolation, y_extrapolation, color='red', label='Extrapolation')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Extrapolation')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
