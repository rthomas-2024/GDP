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


def pid_control(error, prev_error, integral, dt, Kp,Ki,Kd):
    derivative = (error - prev_error) / dt
    integral = np.clip(integral + error * dt, -10, 10)
    return Kp * error + Ki * integral + Kd * derivative, integral


R_Earth = 6378
J2 = 0.00108263

# Target ICs
a = 400 + 6378 # semi major axis
I = np.deg2rad(0)
e = 0
f = np.deg2rad(0)
RAAN = np.deg2rad(0)
AOP = np.deg2rad(0)
mu = 398600 # Earth gravitational param
tau = np.sqrt(a**3 * 4 * np.pi**2 / mu) # orbital period
n = 2*np.pi / tau # mean motion

# Chaser ICs
x0 = 1.5
y0 = 0.4
z0 = 0
dx0 = 0
dy0 = 0
dz0 = 0

# Simulation Parameters
T, dt = 100, 0.1  # Total time and step size
t = np.arange(0, T, dt)
x = np.zeros(len(t))  # Position
dx = np.zeros(len(t))  # Velocity
u_x = np.zeros(len(t))  # Control input
y = np.zeros(len(t))  # Position
dy = np.zeros(len(t))  # Velocity
u_y = np.zeros(len(t))  # Control input
z = np.zeros(len(t))  # Position
dz = np.zeros(len(t))  # Velocity
u_z = np.zeros(len(t))  # Control input

x[0] = x0
y[0] = y0
z[0] = z0
dx[0] = dx0
dy[0] = dy0
dz[0] = dz0

x_ref = 1  # Desired position
y_ref = 0
z_ref = 0
x_err = 0
y_err = 0
z_err = 0
prev_x_err = 0
prev_y_err = 0
prev_z_err = 0
integral_x = 0
integral_y = 0
integral_z = 0

kPx = 3
kIx = 2
kDx = 1
kPy = 2
kIy = 1
kDy = 1
kPz = 0
kIz = 0
kDz = 0

# Simulate PID control
for i in range(1, len(t)):
    x_err = x_ref - x[i-1]
    u_x[i], integral_x = pid_control(x_err, prev_x_err, integral_x, dt, kPx,kIx,kDx)
    prev_x_err = x_err
    y_err = y_ref - y[i-1]
    u_y[i], integral_y = pid_control(y_err, prev_y_err, integral_y, dt, kPy,kIy,kDy)
    prev_y_err = y_err
    z_err = z_ref - z[i-1]
    u_z[i], integral_z = pid_control(z_err, prev_z_err, integral_z, dt, kPz,kIz,kDz)
    prev_z_err = z_err
    
    # System dynamics update
    ddx = u_x[i] + 2*n*dy[i-1] + 3*x[i-1]*n**2
    dx[i] = dx[i-1] + ddx * dt
    x[i] = x[i-1] + dx[i-1] * dt
    ddy = u_y[i] - 2*n*dx[i-1]
    dy[i] = dy[i-1] + ddy * dt
    y[i] = y[i-1] + dy[i-1] * dt
    ddz = u_z[i] - z[i]*n**2
    dz[i] = dz[i-1] + ddz * dt
    z[i] = z[i-1] + dz[i-1] * dt
    
print(x)

# Plot results
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.scatter(0, 0, 0,s=100, marker='.', c='k', label='origin')
ax1.plot(x, y, z, c='g', label="from LVLH" )
ax1.plot(x[0],y[0],z[0],'o',c='r', label='Start')
ax1.plot(x_ref,y_ref,z_ref,'o',c='b', label='Ref')

ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')

#ax1.set_title("lvlh frame, {} orbits".format(round(t/tau,2)))
plt.grid(True)
ax1.legend()


plt.figure(figsize=(8, 4))
plt.plot(t, x,'b', label="Position (x)")
plt.plot(t, dx,'b--', label="Velocity (x)")
plt.axhline(x_ref, color='black', linewidth=0.8, linestyle='--', label="Reference, x")
plt.plot(t, y,'r', label="Position (y)")
plt.plot(t, dy, 'r--',label="Velocity (y)")
plt.axhline(y_ref, color='black', linewidth=0.8, linestyle='--', label="Reference, y")
plt.plot(t, z,'g', label="Position (z)")
plt.plot(t, dz, 'g--',label="Velocity (z)")
plt.axhline(z_ref, color='black', linewidth=0.8, linestyle='--', label="Reference, z")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Closed-Loop System Response with PID Control")
plt.legend()
plt.grid()
plt.show()