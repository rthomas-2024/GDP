import numpy as np
from typing import List

class Actuator:
    def __init__(self,
                 model: str,
                 location: np.ndarray,
                 orientation: np.ndarray): # constructor function
        
        self.model: str = model
        self.location: np.ndarray = location # this is the location of the actautor in the body axis
        self.orientation: np.ndarray = orientation # this is the orientation of the actautor in the body axis

        self.status: bool = False # on or off
        
        # check statements
        if location.shape != (1,3):
            raise ValueError("location must have shape (1, 3)")
        if orientation.shape != (1,3):
            raise ValueError("orientation must have shape (1, 3)")
        
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
                 thrusters: List[Thruster],
                 reactionwheels: List[ReactionWheel]):
        
        self.name: str = name
        self.mass: float = mass
        self.I: np.ndarray = I # inertia matrix
        self.thrusters: List[Thruster] = thrusters
        self.reactionwheels: List[ReactionWheel] = reactionwheels

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