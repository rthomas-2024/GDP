import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odePID(t,u,params):
    c,k,kD,kP,kI = params
    y,v,E = u
    e = yr-y
    if t == 0: e = 0; print(v); print(y); print(v-c*y+kD*e)
    return [ v-c*y+kD*e, kP*e+kI*E-k*y, e ]

# Define system parameters
m = 1.0  # Mass (kg)
c = 0.5  # Damping coefficient (Ns/m)
k = 2.0  # Spring constant (N/m)

# PID Controller Parameters
kP = 10  # Proportional gain
kI = 5   # Integral gain
kD = 3   # Derivative gain

# def pid_control(error, prev_error, integral, dt):
#     derivative = (error - prev_error) / dt
#     integral = np.clip(integral + error * dt, -10, 10)
#     return Kp * error + Ki * integral + Kd * derivative, integral
T = 10
dt = 0.0001
ts = np.arange(0, T+1, dt)
t_span = (0, T+1)

ICs = [0,0,0]

yr = 2

params =[c,k,kD,kP,kI]

sol = solve_ivp(odePID, t_span, ICs, args=[params], t_eval=ts, method='RK45',rtol=1e-10)

ys = sol.y[0]
vs = sol.y[1]
y_s = np.zeros([int((T+1)/dt),1])

for ii in range(0,len(y_s)):
    y_s[ii] = vs[ii] - c*ys[ii] + kD*(yr-ys[ii])
    if ii==0: y_s[ii] = ICs[1]
    

# # Simulation Parameters
# T, dt = 10, 0.01  # Total time and step size
# t = np.arange(0, T, dt)
# x = np.zeros(len(t))  # Position
# v = np.zeros(len(t))  # Velocity
# u = np.zeros(len(t))  # Control input

# x_ref = 1  # Desired position
# error = 0
# prev_error = 0
# integral = 0

# # Simulate PID control
# for i in range(1, len(t)):
#     error = x_ref - x[i-1]
#     u[i], integral = pid_control(error, prev_error, integral, dt)
#     prev_error = error
    
#     # System dynamics update
#     x_ddot = (-k * x[i-1] - b * v[i-1] + u[i]) / m
#     v[i] = v[i-1] + x_ddot * dt
#     x[i] = x[i-1] + v[i-1] * dt

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(ts, sol.y[0], label="Position (x)")
plt.plot(ts, y_s, label = "Velocity (v)")
plt.axhline(yr, color='black', linewidth=0.8, linestyle='--', label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Closed-Loop System Response with PID Control")
plt.legend()
plt.grid()
plt.show()
