import matplotlib.pyplot as plt
import numpy as np

# Define the grid size
W, H = 5, 5

# Define hypothetical coord_q and coord_k (representing cropped regions)
coord_q = np.array([0.2, 0.2, 0.0, 0.7, 0.7, 0.5])
coord_k = np.array([0.3, 0.3, 0.1, 0.8, 0.8, 0.6])

# Define the bin indices for x and y
x_array = np.arange(W)
y_array = np.arange(H)

# Calculate bin widths and heights for coord_q and coord_k
q_bin_width = (coord_q[3] - coord_q[0]) / W
q_bin_height = (coord_q[4] - coord_q[1]) / H
k_bin_width = (coord_k[3] - coord_k[0]) / W
k_bin_height = (coord_k[4] - coord_k[1]) / H

# Calculate the bin centers for coord_q
x_centers_q = coord_q[0] + (x_array + 0.5) * q_bin_width
y_centers_q = coord_q[1] + (y_array + 0.5) * q_bin_height

# Calculate the bin centers for coord_k
x_centers_k = coord_k[0] + (x_array + 0.5) * k_bin_width
y_centers_k = coord_k[1] + (y_array + 0.5) * k_bin_height

# Print bin centers for verification
print(f"x_centers_q: {x_centers_q}")
print(f"y_centers_q: {y_centers_q}")
print(f"x_centers_k: {x_centers_k}")
print(f"y_centers_k: {y_centers_k}")

# Create a meshgrid for plotting
X_q, Y_q = np.meshgrid(x_centers_q, y_centers_q)
X_k, Y_k = np.meshgrid(x_centers_k, y_centers_k)

# Plot the grid and centers
plt.figure(figsize=(8, 8))
plt.plot(X_q, Y_q, 'ro', label='Center q')
plt.plot(X_k, Y_k, 'bx', label='Center k')
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Normalized x')
plt.ylabel('Normalized y')
plt.title('Visualization of Bin Centers in 2D Grid for coord_q and coord_k')
plt.legend()
plt.show()