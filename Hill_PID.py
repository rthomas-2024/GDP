import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odePID(t,u,params):
    n, kPx,kDx,kIx,kPy,kDy,kIy,kPz,kDz,kIz = params
    x,y,z,vx,vy,vz,Ex,Ey,Ez = u
    ex = xr-x; ey = yr-y; ez = zr-z
    dEx=ex; dEy=ey; dEz=ez
    dx = vx + 2*n*y + kDx*dEx
    dy = vy - 2*n*x + kDy*dEy
    dz = vz + kDz*dEz
    dvx = 3*x*n**2 + kPx*dEx + kIx*Ex
    dvy = kPy*dEy + kIy*Ey
    dvz = -z*n**2 + kPz*dEz + kIz*Ez
    
    if t == 0: ex=0;ey=0;ez=0
    
    return [ dx,dy,dz,dvx,dvy,dvz,dEx,dEy,dEz]


# PID Controller Parameters
kPx = 0  # Proportional gain
kIx = 0   # Integral gain
kDx = 0   # Derivative gain
kPy = 0  # Proportional gain
kIy = 0   # Integral gain
kDy = 0   # Derivative gain
kPz = 0  # Proportional gain
kIz = 0   # Integral gain
kDz = 0   # Derivative gain

# Hill eqns set-up:
R_Earth = 6378
J2 = 0.00108263

a = 400 + 6378 # semi major axis
I = np.deg2rad(0)
e = 0
f = np.deg2rad(0)
RAAN = np.deg2rad(0)
AOP = np.deg2rad(0)
mu = 398600 # Earth gravitational param
tau = np.sqrt(a**3 * 4 * np.pi**2 / mu) # orbital period
n = 2*np.pi / tau # mean motion
T = int(1*tau); # time for to dock
ts = np.linspace(0,T,T+1)
t_span = (0, T+1)
dt = 1

# Chaser ICs
x0 = 0.1
y0 = 0.1
z0 = 0
dx0 = 0
dy0 = 0.001
dz0 = 0

ICs = [x0,y0,z0,dx0,dy0,dz0,0,0,0]

xr = 0
yr = 0
zr = 0

params =[n, kPx,kDx,kIx,kPy,kDy,kIy,kPz,kDz,kIz]

sol = solve_ivp(odePID, t_span, ICs, args=[params], t_eval=ts, method='RK45',rtol=1e-10)

xs = sol.y[0]
ys = sol.y[1]
zs = sol.y[2]
vxs = sol.y[3]
vys = sol.y[4]
vzs = sol.y[5]

dxs = np.zeros([int((T+1)/dt),1])
dys = np.zeros([int((T+1)/dt),1])
dzs = np.zeros([int((T+1)/dt),1])

for ii in range(0,len(ys)):
    dxs[ii] = vxs[ii] + 2*n*ys[ii] + kDx*(xr-xs[ii])
    dys[ii] = vys[ii] - 2*n*xs[ii] + kDy*(yr-ys[ii])
    dzs[ii] = vzs[ii] + kDz*(zr-zs[ii])
    if ii==0: dxs[ii] = ICs[3]; dys[ii] = ICs[4]; dzs[ii] = ICs[5]
    


fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.scatter(0, 0, 0,s=100, marker='.', c='k', label='origin')
ax1.plot(xs, ys, zs, c='g' )

ax1.set_xlabel('x (r-bar), m')
ax1.set_ylabel('y (v-bar), m')

ax1.set_title("lvlh frame, {} orbits".format(round(T/tau,2)))
plt.grid(True)
ax1.legend()

plt.show()
# # Plot results
# plt.figure(figsize=(8, 4))
# plt.plot(ts, sol.y[0], label="Position (x)")
# plt.plot(ts, y_s, label = "Velocity (v)")
# plt.axhline(yr, color='black', linewidth=0.8, linestyle='--', label="Reference")
# plt.xlabel("Time (s)")
# plt.ylabel("Position")
# plt.title("Closed-Loop System Response with PID Control")
# plt.legend()
# plt.grid()
# plt.show()

