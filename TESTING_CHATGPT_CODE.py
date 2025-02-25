import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation dy/dt = -k*y
def decay(t, y, k):
    return -k * y

# Parameters
k = 0.5   # Decay constant
y0 = [1.0]  # Initial condition
t_step = 0.1  # Time step in seconds
t_max = 5.0  # Total simulation time

# Time tracking
t_current = 0.0
start_time = time.perf_counter()

# Lists to store results for plotting
t_values = [0.0]  # Include the initial time point
y_values = [y0[0]]  # Include the initial condition

while t_current < t_max:
    next_time = start_time + t_current + t_step

    # Solve the ODE for one step
    sol = solve_ivp(decay, [t_current, t_current + t_step], y0, args=(k,), method='RK45')

    # Extract the new value
    y0 = [sol.y[0, -1]]  # Update initial condition for next step
    t_current += t_step

    # Store results for plotting
    t_values.append(t_current)
    y_values.append(y0[0])

    # Print results
    print(f"Time: {t_current:.2f}, y: {y0[0]:.6f}")

    # Active waiting to sync with real time
    while time.perf_counter() < next_time:
        pass

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t_values, y_values, marker='o', linestyle='-', label=r"$\frac{dy}{dt} = -ky$")
plt.scatter(0, y_values[0], color='red', zorder=5, label='Initial Condition')  # Plot initial condition
plt.xlabel("Time (s)")
plt.ylabel("y")
plt.title("Real-Time Numerical Solution of ODE")
plt.legend()
plt.grid()
plt.show()
