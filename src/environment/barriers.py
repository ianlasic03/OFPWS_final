import gymnasium
import numpy as np
import pyrorl
#from src.pyrorl_custom.pyro_custom import WildfireEvacuationEnv

class Barriers():
    def __init__(self, env, populated_areas, num_barriers):
        """
        Initialize the environment  with the given parameters.
        """
        self.num_barriers = num_barriers
        self.fire_env = env
        self.populated_areas = populated_areas
        self.num_barriers = num_barriers
        self.barriers = set()

    def add_barrier(self) -> set:
        """
        At the begining of the simualtion, add a barrier to the environment.
        Choose a random cell in the environment, get it's state space, and update it's fuel_index to 0 

        Args:
            Takes in the created environment !! Have a created environment before calling this method
            env (gymnasium.Env): The environment to which the barrier will be added.
            barrier (list): List of coordinates representing the barrier.
        """
        # Get the shape of the environment
        fire_env = self.fire_env
        # Get the state space of the environment
        state_space = fire_env.get_state()
        (_, rows, cols) = state_space.shape
        populated = self.populated_areas
        
        # Randomly choose barrier states
        while len(self.barriers) < self.num_barriers:
            # Randomly choose a barrier state
            barrier = tuple(np.random.randint(0, [rows, cols])) # Should produce a cell (x, y) in the environment
            
            # Check if the barrier is not in the populated areas
            if (not np.any(np.all(barrier == populated, axis=1))) and (barrier not in self.barriers):
                # Set the fuel index to 0 for that cell
                state_space[1][barrier[0]][barrier[1]] = 0  # 1 = FUEL_INDEX
                # Add the barrier to the set of barriers
                self.barriers.add(barrier) 
        
        return self.barriers
        