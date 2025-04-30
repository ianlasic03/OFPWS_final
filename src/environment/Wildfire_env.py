import gymnasium
import numpy as np
import pyrorl

class WildfireEnv():
    def __init__(self, num_rows, num_cols, populated_areas, paths, paths_to_pops):
        """
        Initialize the environment with the given parameters.

        Args:
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
            populated_areas (np.ndarray): Array of populated areas.
            paths (np.ndarray): Array of paths.
            paths_to_pops (dict): Dictionary mapping paths to populated areas.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.populated_areas = populated_areas
        self.paths = paths
        self.paths_to_pops = paths_to_pops
        
    def create_env(self):
    # Initialize the environment
        kwargs = {
        "num_rows": self.num_rows,
        "num_cols": self.num_cols,
        "populated_areas": self.populated_areas,
        "paths": self.paths,
        "paths_to_pops": self.paths_to_pops,
        }
        return gymnasium.make("pyrorl/PyroRL-v0", **kwargs)
    
    def add_barrier(self, env, num_barriers) -> None:
        """
        !!Currently doesn't work!!
        
        Add a barrier to the environment.

        Args:
            Takes in the created environment !! Have a created environment before calling this method
            env (gymnasium.Env): The environment to which the barrier will be added.
            barrier (list): List of coordinates representing the barrier.
        """
        # Get the shape of the environment
        fire_env = env.unwrapped
        num_rows, num_cols = fire_env.state_space.shape[1], fire_env.state_space.shape[2]
        populated = self.populated_areas

        barriers = set()

        # Randomly choose barrier states
        while len(barriers) < num_barriers:
            # Randomly choose a barrier state
            barrier = tuple(np.random.randint(0, [num_rows, num_cols]))
            # Check if the barrier is not in the populated areas
            if (not np.any(np.all(barrier == populated, axis=1))) and (barrier not in barriers):
                # Maybe have to turn fire index to zero as well?
                fire_env.state_space[1, barrier[0], barrier[1]] = 0  # 1 = FUEL_INDEX
                barriers.add(barrier) 
        
            
        