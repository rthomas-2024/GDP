import numpy as np
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Actuator:
    def __init__(self,
                 model: str,
                 location: np.ndarray,
                 orientation: np.ndarray,
                 F_max: float): # constructor function
        
        self.model: str = model
        self.location: np.ndarray = location # this is the location of the actautor in the body axis
        self.orientation: np.ndarray = orientation # this is the orientation of the actautor in the body axis
        self.STATE = False # bool, if the thruster is on/on. initiate as off (False)
        self.F_max: float = F_max
        self.status: bool = False # on or off
        
        # check statements
        if location.shape != (1,3):
            raise ValueError("location must have shape (1, 3)")
        if orientation.shape != (1,2):
            raise ValueError("orientation must have shape (1, 2)")
        
    def __str__(self):
        return (f"Model Name: {self.model}, Location: {self.location.flatten()}, "
                f"Orientation: {self.orientation.flatten()}")
     
class Thruster(Actuator):
    def ShowType(self):
        print(" This is a thruster")
     
class ReactionWheel(Actuator):
    def ShowType(self):
        print(" This is a reaction wheel")
       
class SpaceCraft:
    def __init__(self, 
                 name: str,
                 mass: float,
                 I: np.ndarray,
                 SC_shape: np.ndarray,
                 thrusters: List[Thruster],
                 reactionwheels: List[ReactionWheel]):
        
        self.name: str = name
        self.mass: float = mass
        self.I: np.ndarray = I # inertia matrix
        self.SC_shape: np.ndarray = SC_shape
        self.thrusters: List[Thruster] = thrusters
        self.reactionwheels: List[ReactionWheel] = reactionwheels
        
        self.ThrustVectorMatrix: np.array = np.zeros([len(self.thrusters),6])
        print(self.ThrustVectorMatrix)
        for ii in range(0,len(self.thrusters)):
            T_ii = self.thrusters[ii]
            az, el = np.deg2rad(T_ii.orientation[0,0]), np.deg2rad(T_ii.orientation[0,1])
            x_b, y_b, z_b = T_ii.location[0] # body frame coordinate
            unit_vector = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el)
                    ])
            Fx_ii, Fy_ii, Fz_ii = T_ii.F_max*unit_vector
            self.ThrustVectorMatrix[ii,0:3] = Fx_ii, Fy_ii, Fz_ii
            tau_ii = np.cross(np.array([x_b,y_b,z_b]), np.array([Fx_ii,Fy_ii,Fz_ii])) # calculate the torque using the right hand rule
            self.ThrustVectorMatrix[ii,3:6] = tau_ii

                


        if I.shape != (3,3):
            raise ValueError("I must have shape (3, 3)")
        
        
    def __str__(self):
        # Basic spacecraft information
        spacecraft_info = (f"SpaceCraft Name: {self.name}\n"
                           f"Mass: {self.mass} kg\n"
                           f"Inertia Matrix:\n{self.I}\n")
        
        # Thruster information
        spacecraft_info += f"\nNumber of Thrusters: {len(self.thrusters)}\n"
        for i, thruster in enumerate(self.thrusters, start=1):
            spacecraft_info += f"  Thruster {i}..... {thruster}\n"

        # Reaction Wheel information
        spacecraft_info += f"\nNumber of Reaction Wheels: {len(self.reactionwheels)}\n"
        for i, wheel in enumerate(self.reactionwheels, start=1):
            spacecraft_info += f"  Reaction Wheel {i}..... {wheel}\n"
    
        return spacecraft_info 
    
    def DrawSpaceCraft(self, DRAW_THRUSTERS=0):
        b, h, w = self.SC_shape[0]/2, self.SC_shape[1]/2, self.SC_shape[2]/2

        # Define the 8 vertices of the cuboid
        vertices = np.array([
            [-b, -h, -w], [b, -h, -w], [b, h, -w], [-b, h, -w],  # Front face
            [-b, -h, w], [b, -h, w], [b, h, w], [-b, h, w]       # Back face
        ])

        # Define the 6 faces using the vertices
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],  # Front face
            [vertices[j] for j in [4, 5, 6, 7]],  # Back face
            [vertices[j] for j in [0, 1, 5, 4]],  # Bottom face
            [vertices[j] for j in [2, 3, 7, 6]],  # Top face
            [vertices[j] for j in [0, 3, 7, 4]],  # Left face
            [vertices[j] for j in [1, 2, 6, 5]]   # Right face
        ]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Add faces to plot
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.3))
        
        # Ensure the axes have equal scale
        max_range = 1.5 * max(self.SC_shape) / 2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect([1, 1, 1])
        # Labels
        ax.set_xlabel('X, m')
        ax.set_ylabel('Y, m')
        ax.set_zlabel('Z, m')
        # Draw the XYZ axes at the centroid (0, 0, 0)
        ax.quiver(0, 0, 0, 1, 0, 0, color='k')  # X axis
        ax.quiver(0, 0, 0, 0, 1, 0, color='k')  # Y axis
        ax.quiver(0, 0, 0, 0, 0, 1, color='k')  # Z axis

        # Label the axes heads
        ax.text(1.1, 0, 0, 'X', color='k', fontsize=12, weight='bold')
        ax.text(0, 1.1, 0, 'Y', color='k', fontsize=12, weight='bold')
        ax.text(0, 0, 1.1, 'Z', color='k', fontsize=12, weight='bold')
        
        if DRAW_THRUSTERS == 1:
            for ii in range(0,len(self.thrusters)):
                T_ii = self.thrusters[ii]
                az, el = np.deg2rad(T_ii.orientation[0,0]), np.deg2rad(T_ii.orientation[0,1])
                # # Flip direction if azimuth > 180 degrees
                # if az > np.pi:
                #     az -= np.pi  # Subtract 180 degrees (pi radians)
                #     el = -el  # Flip the elevation to mirror the vector
                x_b, y_b, z_b = T_ii.location[0] # body frame coordinate
                unit_vector = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el)
                     ])
                # Plot the original point
                random_color = np.random.rand(3,)  # Random values between 0 and 1
                ax.plot(x_b,y_b,z_b,'o',markersize=4, color='k')

                # Plot the vector
                ax.quiver(*np.array([x_b,y_b,z_b]), *unit_vector, color=random_color, length=1, linewidth=2, arrow_length_ratio=0.1,label=T_ii.model)
                ax.text(x_b*1.1, y_b*1.1, z_b*1.1, T_ii.model[-1], color='k', fontsize=12, weight='bold')

        plt.legend()
        plt.show()
        
