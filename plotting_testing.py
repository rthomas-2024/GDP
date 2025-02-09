import numpy as np
import matplotlib.pyplot as plt

def unit_vector_from_azimuth_elevation(azimuth, elevation):
    """
    Calculate unit vector from azimuth and elevation, flipping direction if az > 180.
    
    Args:
    - azimuth: Azimuth angle in degrees (0 to 360)
    - elevation: Elevation angle in degrees (-90 to +90)
    
    Returns:
    - unit_vector: Corresponding 3D unit vector
    """
    # Convert angles to radians
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    
    # Flip direction if azimuth > 180 degrees
    if azimuth > np.pi:
        azimuth -= np.pi  # Subtract 180 degrees (pi radians)
        elevation = -elevation  # Flip the elevation to mirror the vector

    # Calculate the unit vector components
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    
    return np.array([x, y, z])

# Example usage
azimuth = 270  # Azimuth greater than 180 (should flip direction)
elevation = 45  # Elevation (in degrees)

# Calculate the unit vector
unit_vec = unit_vector_from_azimuth_elevation(azimuth, elevation)

print("Unit Vector:", unit_vec)

# Plotting the unit vector in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the unit vector
ax.quiver(0, 0, 0, unit_vec[0], unit_vec[1], unit_vec[2], color='b', length=1)

# Setting the aspect ratio and limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Labeling the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display the plot
plt.show()
