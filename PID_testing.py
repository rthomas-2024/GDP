import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
m = 1.0  # Mass (kg)
b = 0.5  # Damping coefficient (Ns/m)
k = 2.0  # Spring constant (N/m)

# Define state-space matrices
A = np.array([[0, 1],
              [-k/m, -b/m]])
B = np.array([[0],
              [1/m]])
C = np.array([[1, 0]])  # Measure position only
D = np.array([[0]])

# PID Controller Parameters
Kp = 10  # Proportional gain
Ki = 5   # Integral gain
Kd = 3   # Derivative gain

def pid_control(error, prev_error, integral, dt):
    derivative = (error - prev_error) / dt
    integral = np.clip(integral + error * dt, -10, 10)
    return Kp * error + Ki * integral + Kd * derivative, integral

# Simulation Parameters
T, dt = 10, 0.0001  # Total time and step size
t = np.arange(0, T, dt)
x = np.zeros(len(t))  # Position
v = np.zeros(len(t))  # Velocity
u = np.zeros(len(t))  # Control input

x_ref = 2  # Desired position
error = 0
prev_error = 0
integral = 0

# Simulate PID control
for i in range(1, len(t)):
    error = x_ref - x[i-1]
    u[i], integral = pid_control(error, prev_error, integral, dt)
    prev_error = error
    
    # System dynamics update
    x_ddot = (-k * x[i-1] - b * v[i-1] + u[i]) / m
    v[i] = v[i-1] + x_ddot * dt
    x[i] = x[i-1] + v[i-1] * dt

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(t, x, label="Position (x)")
plt.plot(t, v, label="Velocity (v)")
plt.axhline(x_ref, color='black', linewidth=0.8, linestyle='--', label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Closed-Loop System Response with PID Control")
plt.legend()
plt.grid()
plt.show()

print(v)