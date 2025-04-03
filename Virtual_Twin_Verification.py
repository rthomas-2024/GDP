import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def readCSV():
    df = pd.read_csv("VT_Output_Traj/VT_Data_1.5-1.csv")
    data = df.to_numpy()

    ts = data[:,0]
    xs, ys, zs = data[:,1], data[:,2], data[:,3]
    qs = data[:,4:8]
    joints = data[:,8:14]


    return ts, xs, ys, zs, qs, joints


ts, xs, ys, zs, qs, joints = readCSV()
rs, ps, yws = 


fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)  # 6 rows, 1 column, shared x-axis

# Titles for each subplot
titles = [
    "Y-Axis Position",
    "Z-Axis Position",
    "Gimbal Holder Position",
    "Pitch Axis Position",
    "Roll Axis Position",
    "End Effector Interface Position"]

#Display rotations in degrees not radians
pitchAxisPosPlt = np.rad2deg(pitchAxisPosPlt)
rollAxisPosPlt = np.rad2deg(rollAxisPosPlt)
endEffInterfacePosPlt = np.rad2deg(endEffInterfacePosPlt)

# Data for each subplot
positions = [yaxisPosPlt, zaxisPosPlt, gimbalHolderPosPlt, pitchAxisPosPlt, rollAxisPosPlt, endEffInterfacePosPlt]

# Plot each joint position on a separate axis
for i, ax in enumerate(axes):
    ax.plot(ts, positions[i], label=titles[i], color='b')
    ax.set_ylabel("Position")
    ax.set_title(titles[i])
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Time (s)")  # X-axis label only on the last plot

plt.tight_layout()  # Adjust spacing
plt.show()


#plot end eff link position
#unpack position data
endEff_xs, endEff_ys, endEff_zs = zip(*endEffPosPlt)
endEff_xs, endEff_ys, endEff_zs = list(endEff_xs), list(endEff_ys), list(endEff_zs)


#plot position data
plt.figure()
plt.plot(endEff_xs, endEff_ys)
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid()
plt.show()