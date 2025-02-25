import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the inertia tensor (general case, non-diagonal)
I = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])  # Example inertia tensor (kg*m^2)

# Define the inverse of the inertia tensor for easier computation
I_inv = np.linalg.inv(I)

# Define external torque (can be time-dependent or constant)
def external_torque(t):
    return np.array([0.0, 0.0, 0.0])  # No external torques for now (free motion)

# Euler's equations of motion in full form
def euler_equations(t, omega):
    omega = np.array(omega)
    torque = external_torque(t)
    
    # Compute angular momentum H = I * omega
    H = I @ omega
    
    # Compute d(omega)/dt = I^(-1) * (M - omega x H)
    omega_dot = I_inv @ (torque - np.cross(omega, H))
    
    return omega_dot

# Initial angular velocity (rad/s) in body frame
omega_0 = np.array([0,0,np.deg2rad(1)])

# Time span for integration (in seconds)
t_span = (0, 60)  # Simulate for 10 seconds
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Solve the system using SciPy's solve_ivp
solution = solve_ivp(euler_equations, t_span, omega_0, t_eval=t_eval, method='RK45')

# Extract results
t_vals = solution.t  # Time values
omega_vals = solution.y  # Angular velocity components over time

print(omega_vals)

plt.plot(omega_vals[0,:],t_eval, "b", label="x")
plt.plot(omega_vals[1,:],t_eval, "r", label="y")
plt.plot(omega_vals[2,:],t_eval, "g", label="z")
plt.grid()
plt.legend()
plt.show()