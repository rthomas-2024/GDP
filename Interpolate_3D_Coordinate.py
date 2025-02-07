import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate_3d(points, times, t_query):
    """
    Interpolates the 3D coordinates at a given time t_query.

    :param points: List of (x, y, z) tuples.
    :param times: List of timestamps corresponding to each point.
    :param t_query: The target time for interpolation.
    :return: Interpolated (x, y, z) coordinates at t_query.
    """
    times = np.array(times)
    points = np.array(points)  # Shape (N, 3)
    
    # Create interpolation functions for x, y, z
    interp_x = interp1d(times, points[:, 0], kind='cubic')
    interp_y = interp1d(times, points[:, 1], kind='cubic')
    interp_z = interp1d(times, points[:, 2], kind='cubic')

    # Interpolate at t_query
    x_t = interp_x(t_query)
    y_t = interp_y(t_query)
    z_t = interp_z(t_query)

    return np.array([x_t, y_t, z_t])

# Example usage
points = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
times = [0, 5, 10]
t_query = 7  # Query at t=7

interpolated_point = interpolate_3d(points, times, t_query)
print("Interpolated Point:", interpolated_point)

# Plot the trajectory and interpolated point
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo-', label='Trajectory')
ax.plot(interpolated_point[0], interpolated_point[1],interpolated_point[2], 'r.', label=f'Interpolated t={t_query}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title("3D Trajectory Interpolation")
plt.show()