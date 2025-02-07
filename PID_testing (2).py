import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system parameters
a, b = 1.0, 1.0  # Example system parameters
Kp, Ki, Kd = 3, 10, 1  # PID gains

# Reference setpoint
x_ref = 1.0

# PID state variables
integral = 0
previous_error = 0
previous_time = 0
print("start")
# PID controller function
def pid_control(t, error):
    global integral, previous_error, previous_time
    
    # Calculate time step dynamically
    dt = t - previous_time 
    print(dt)
    integral += error * dt
    derivative = (error - previous_error) / dt if dt > 0 else 0
    previous_error = error
    previous_time = t
    
    return Kp * error + Ki * integral + Kd * derivative

# System dynamics with PID control
def system_dynamics(t, x):
    error = x_ref - x[0]
    u = pid_control(t, error)
    dxdt = -a * x[0] + b * u
    return [dxdt]

# Solve ODE using solve_ivp
dt = 0.1
t_span = (0, 10)  # Time range
t_eval = np.arange(t_span[0], t_span[1]+dt, dt) # when to store state matrix

x0 = [0]  # Initial state

solution = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, method="RK45")

# Plot results
plt.plot(solution.t, solution.y[0], label="System Response")
plt.axhline(x_ref, color="r", linestyle="--", label="Reference")
plt.xlabel("Time")
plt.ylabel("State (x)")
plt.legend()
plt.title("PID Control using solve_ivp with Dynamic Time Step")
plt.show()