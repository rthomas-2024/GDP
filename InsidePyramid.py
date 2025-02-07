import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def is_inside_2d(edges, xp, yp):
    """Checks if a 2D point (xp, yp) is inside a polygon defined by edges using ray-casting."""
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1

def check_points_in_pyramid(points: np.ndarray, base, apex) -> np.ndarray:
    """
    Checks which points are inside a 3D pyramid.

    Parameters:
    - points (np.ndarray): NumPy array of shape (N, 3) with (x, y, z) points.
    - base (list): List of (x, y, z) tuples defining the base of the pyramid.
    - apex (tuple): (x, y, z) coordinates of the pyramid's apex.

    Returns:
    - NumPy array (N, 4), where the first three columns are (x, y, z),
      and the fourth column is 1 (inside) or 0 (outside).
    """
    if points.shape[1] != 3:
        raise ValueError("Input points array must have shape (N, 3)")

    # Define 2D projections
    xy_triangle = [(base[0][0], base[0][1]), (base[1][0], base[1][1]), (apex[0], apex[1])]
    zx_triangle = [(base[1][0], base[1][2]), (base[2][0], base[2][2]), (apex[0], apex[2])]

    # Convert to edge lists for 2D checking
    xy_edges = list(zip(xy_triangle, xy_triangle[1:] + [xy_triangle[0]]))
    zx_edges = list(zip(zx_triangle, zx_triangle[1:] + [zx_triangle[0]]))

    # Initialize results array with an extra column for inside/outside check
    results = np.zeros((points.shape[0], 4))
    results[:, :3] = points  # Copy x, y, z values

    # Check each point
    for i, (x, y, z) in enumerate(points):
        inside_xy = is_inside_2d(xy_edges, x, y)
        inside_zx = is_inside_2d(zx_edges, x, z)
        results[i, 3] = int(inside_xy and inside_zx)  # Store 1 (inside) or 0 (outside)

    return results

def plot_pyramid_with_points(base, apex, results):
    """
    Plots a 3D pyramid and its 2D projections (XY and ZX) with the given points.
    
    Parameters:
    - base: List of (x, y, z) tuples defining the base.
    - apex: (x, y, z) coordinates of the pyramid's apex.
    - results: NumPy array containing (x, y, z, inside) points.
    """
    # Define pyramid faces
    faces = [
        [base[0], base[1], apex],  # Front face
        [base[1], base[2], apex],  # Right face
        [base[2], base[3], apex],  # Back face
        [base[3], base[0], apex],  # Left face
        base  # Base
    ]

    # Extract 2D projections
    xy_triangle = [(base[0][0], base[0][1]), (base[1][0], base[1][1]), (apex[0], apex[1])]
    zx_triangle = [(base[1][0], base[1][2]), (base[2][0], base[2][2]), (apex[0], apex[2])]

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # ---- 3D Plot ----
    ax = fig.add_subplot(131, projection='3d')
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, edgecolor='k'))

    # Plot points
    for x, y, z, inside in results:
        ax.scatter(x, y, z, color='g' if inside else 'r', s=50)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Pyramid")
    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 6])
    ax.set_zlim([-1, 5])

    # ---- XY Projection ----
    ax2 = fig.add_subplot(132)
    xy_xs, xy_ys = zip(*xy_triangle)
    ax2.fill(xy_xs, xy_ys, "b", alpha=0.5)

    for x, y, _, inside in results:
        ax2.scatter(x, y, color='g' if inside else 'r', s=50)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("XY Projection")
    ax2.set_xlim([-1, 5])
    ax2.set_ylim([-1, 6])
    ax2.set_aspect("equal")

    # ---- ZX Projection ----
    ax3 = fig.add_subplot(133)
    zx_xs, zx_zs = zip(*zx_triangle)
    ax3.fill(zx_xs, zx_zs, "r", alpha=0.5)

    for x, _, z, inside in results:
        ax3.scatter(x, z, color='g' if inside else 'r', s=50)

    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("ZX Projection")
    ax3.set_xlim([-1, 5])
    ax3.set_ylim([-1, 5])
    ax3.set_aspect("equal")

    plt.tight_layout()
    plt.show()

# --- Example Usage ---
base = [(0, 1, 1), (0, 5, 1), (0, 5, 5), (0, 1, 5)]  # Base parallel to YZ plane
apex = (4, 3, 3)  # Vertex at (4,3,3)

# Sample points as a NumPy array (N, 3)
points_to_check = np.array([
    [2, 2, 2],  # Inside
    [3, 3, 3],  # Inside
    [4, 3, 3],  # On the apex (inside)
    [1, 4, 4],  # Inside
    [5, 5, 5],  # Outside
    [0, 1, 1],  # On the base (outside)
    [0, 3, 3],  # On the base (outside)
    [2, 5, 1],  # Inside
    [0, 6, 6],  # Outside
])

# Run check
results = check_points_in_pyramid(points_to_check, base, apex)

# Display results
print("Checked Points:\n", results)

# Plot the results
plot_pyramid_with_points(base, apex, results)
