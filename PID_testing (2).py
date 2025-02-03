import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def compute(self, measured_value, dt):
        error = self.setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class MotionController:
    def __init__(self, n, dt):
        self.n = n
        self.dt = dt

        # PID Controllers for x, y, and z
        self.pid_x = PIDController(1.0, 0.1, 0.05)
        self.pid_y = PIDController(1.0, 0.1, 0.05)
        self.pid_z = PIDController(1.0, 0.1, 0.05)

        # Initialize state variables
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0

    def update(self, fx_setpoint, fy_setpoint, fz_setpoint):
        # Set setpoints for the PID controllers
        self.pid_x.setpoint = fx_setpoint
        self.pid_y.setpoint = fy_setpoint
        self.pid_z.setpoint = fz_setpoint

        # Compute control forces using PID
        fx = self.pid_x.compute(self.x, self.dt)
        fy = self.pid_y.compute(self.y, self.dt)
        fz = self.pid_z.compute(self.z, self.dt)

        # Update accelerations
        ax = fx - 2 * self.n * self.vy - 3 * self.n**2 * self.x
        ay = fy + 2 * self.n * self.vx
        az = fz + self.n**2 * self.z

        # Update velocities
        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.vz += az * self.dt

        # Update positions
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.z += self.vz * self.dt

        return self.x, self.y, self.z

# Main simulation with 3D plotting
if __name__ == "__main__":
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

    dt = 0.01  # Time step
    controller = MotionController(n, dt)

    # Desired setpoints for fx, fy, fz
    fx_setpoint = 0
    fy_setpoint = 0
    fz_setpoint = 0

    # Simulation parameters
    simulation_time = 10  # seconds
    time_steps = int(simulation_time / dt)

    # Data storage for plotting
    x_vals = []
    y_vals = []
    z_vals = []

    # Simulate motion
    for _ in range(time_steps):
        x, y, z = controller.update(fx_setpoint, fy_setpoint, fz_setpoint)

        # Store data
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    # 3D Plotting the trajectory
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_vals, y_vals, z_vals, label="Trajectory")
    ax.set_title("3D Position Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.legend()
    ax.grid()

    # Show the 3D plot
    plt.show()
