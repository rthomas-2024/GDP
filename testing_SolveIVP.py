import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation dy/dt = f(t, y)
def f(t, y):
    return -2 * t * y  # Example: dy/dt = -2ty

# Initial conditions
t0 = 0      # Initial time
y0 = [1]    # Initial value of y
t_end = 5   # Final time

# Define the times at which to get the solution
num_points = 100  # Number of evenly distributed points
t_eval = np.linspace(t0, t_end, num_points)

# Solve the IVP
solution = solve_ivp(f, (t0, t_end), y0, t_eval=t_eval, method='RK45')

# Extract the results
t = solution.t       # Times at which the solution was evaluated
y = solution.y[0]    # Corresponding values of y

# Display the solution
plt.plot(t, y, label="y(t)")
plt.xlabel("Time (t)")
plt.ylabel("Solution (y)")
plt.title("Solution of the IVP")
plt.legend()
plt.grid()
plt.show()

# Print the results
print("Times:", t)
print("Solution:", y)
