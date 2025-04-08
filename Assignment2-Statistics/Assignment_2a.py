import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 26560e3  # semi-major axis in meters
e = 0.01  # eccentricity
c = 299792458  # speed of light in m/s
mu = 3.986e14  # gravitational parameter in m^3/s^2

# True anomaly (θ) in radians
theta = np.linspace(0, 2 * np.pi, 1000)

# Orbital radius (r)
r = a * (1 - e**2) / (1 + e * np.cos(theta))



# Orbital velocity (v)
v = np.sqrt(mu*(1 + 2 * e * np.cos(theta) + e**2) / (a * (1 - e**2)))



# Flight path angle (γ)
gamma = np.arctan(e * np.sin(theta) / (1 + e * np.cos(theta)))



# Relativistic correction to time delay δt_rel
delta_t_rel = -2 * (r * v * np.cos(np.pi / 2 + gamma)) / c**2

# Correction to pseudo-range
delta_pseudo_range = c * delta_t_rel

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta), delta_pseudo_range)
plt.title("Relativistic Correction vs True Anomaly", fontsize=14)
plt.xlabel("True Anomaly (degrees)", fontsize=12)
plt.ylabel("Relativistic Correction to Pseudo-range (m)", fontsize=12)
plt.grid(True)
plt.savefig("relativistic_correction_vs_true_anomaly.png", dpi=300)
plt.close()


print(delta_pseudo_range.max())


