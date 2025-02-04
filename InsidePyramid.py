import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def is_inside(edges, xp, yp):
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1

# Define the 3D square-based pyramid
base = [(1, 1, 0), (5, 1, 0), (5, 5, 0), (1, 5, 0)]
apex = (3, 3, 4)
faces = [
    [base[0], base[1], apex],  # Front face (XZ plane)
    [base[1], base[2], apex],  # Right face (YZ plane)
    [base[2], base[3], apex],  # Back face
    [base[3], base[0], apex],  # Left face
    base  # Base square
]

# Extract the two 2D triangular faces
xz_triangle = [(base[0][0], base[0][2]), (base[1][0], base[1][2]), (apex[0], apex[2])]
yz_triangle = [(base[1][1], base[1][2]), (base[2][1], base[2][2]), (apex[1], apex[2])]

# Predefined list of points to check
points_to_check = [
    (2, 2, 1),  # Inside
    (4, 2, 3),  # Outside
    (3, 3, 2),  # Inside
    (1, 1, 0),  # On the base (outside)
    (5, 5, 4),  # Inside
]

# Initialize lists to store results
coordinates = []
results = []

# Check each point from the predefined list
for x_input, y_input, z_input in points_to_check:
    # Check if inside the 2D projections
    inside_xz = is_inside(list(zip(xz_triangle, xz_triangle[1:] + [xz_triangle[0]])), x_input, z_input)
    inside_yz = is_inside(list(zip(yz_triangle, yz_triangle[1:] + [yz_triangle[0]])), y_input, z_input)

    coordinates.append((x_input, y_input, z_input))
    results.append(inside_xz and inside_yz)

# Plot the 3D pyramid
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(131, projection='3d')
ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, edgecolor='k'))

# Plot the points and indicate whether inside or outside
for coord, result in zip(coordinates, results):
    x, y, z = coord
    ax.scatter(x, y, z, color='g' if result else 'r', s=50)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Pyramid")
ax.set_xlim([0, 6])
ax.set_ylim([0, 6])
ax.set_zlim([0, 5])

# Plot the XZ triangular face
ax2 = fig.add_subplot(132)
xz_xs, xz_zs = zip(*xz_triangle)
ax2.fill(xz_xs, xz_zs, "b", alpha=0.5)
for coord, result in zip(coordinates, results):
    x, _, z = coord
    ax2.scatter(x, z, color='g' if result else 'r', s=50)
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
ax2.set_title("XZ Face")
ax2.set_xlim([0, 6])
ax2.set_ylim([0, 5])
ax2.set_aspect("equal")

# Plot the YZ triangular face
ax3 = fig.add_subplot(133)
yz_ys, yz_zs = zip(*yz_triangle)
ax3.fill(yz_ys, yz_zs, "r", alpha=0.5)
for coord, result in zip(coordinates, results):
    _, y, z = coord
    ax3.scatter(y, z, color='g' if result else 'r', s=50)
ax3.set_xlabel("Y")
ax3.set_ylabel("Z")
ax3.set_title("YZ Face")
ax3.set_xlim([0, 6])
ax3.set_ylim([0, 5])
ax3.set_aspect("equal")

plt.tight_layout()
plt.show()
