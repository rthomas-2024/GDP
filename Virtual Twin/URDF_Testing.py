import numpy as np
import pybullet as p

p.connect(p.GUI)
p.setGravity(0,0,-9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #not showing GUI to make animations look nicer

gantry = p.loadURDF("Gantry_CAD.urdf", basePosition=np.array([0,0,0]))
p.setRealTimeSimulation(1)

