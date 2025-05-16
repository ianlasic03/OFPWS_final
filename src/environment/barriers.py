import gymnasium
import numpy as np
import pyrorl


class Barriers():
    def __init__(self, env, paths, populated_areas, num_barriers):
        """
        Initialize the environment  with the given parameters.
        """
        self.env = env # WildfireEvacuationEnv
        self.world = env.fire_env      # raw FireWorld
        self.paths = paths
        self.populated_areas = populated_areas
        self.num_barriers = num_barriers
        self.barriers = set()

    def simulated_annealing(self, temperature, cooling_rate, kmax):
        """
        Simulated annealing to find the best barrier placement.
        Takes in a 
        - objective function to minimize: total spread of fire -> from env.burned_area()
        - initial state: random placement of barriers (initial state of barriers?)
        - Transition distribution/function: I think this is ->sample_fire_propagation function
        - temperature: initial temperature
        - cooling rate: rate at which the temperature decreases
        - number of iterations: number of iterations to run the algorithm

        Keep track of x_prime and F_prime (best state and fire spread)
        This is confusing, do we keep track of the best state
        OR
        do we keep track of the best placement of barriers
        """
        #wildfire_env = self.env

        # Initial random barrier placement
        init_barriers = self.add_barrier()  # Add barriers to the environment
        # Get fire spread for random barriers (F)
        wildfire_env = self.env
        # This runs through an entire simulation of the environment with initial barriers
        F = wildfire_env.objective(init_barriers)
        #F = fire_env.burned_area()  # Get the initial fire spread
        best_b, best_F = init_barriers, F
        
        for _ in range(kmax):
            print("Current iteration: ", _)
            # get this with the env.burned_area() function in pyro_custom.py
            # Sample a transition from initial state x to new state x' with sample_fire_propagation
            # I'm not sure this will return a new set of barriers the way it's written
            # I think It will return a new state space with the same init_barriers in it
            # Sample a new set of barriers 
            #barriers_prime = self.sample_fire_propagation(init_barriers) # !!!!
            
            # Propose a new set of barriers
            barriers_prime = wildfire_env.propose(best_b)
            # I'm not sure fire_env is actually changed when running propose?
            # Get objective value for new set of barriers
            F_prime = wildfire_env.objective(barriers_prime) # This will reset environment with new barriers and run simulation

            # Calculate the fire spread F' for the new state x'
            # if F' < F, update x = x' and F = F' 
            # F' less than F -> F' is better
            score_diff = F_prime - best_F

            # If score_diff is less than 0, new state is better so accept it
            # if score_ diff is greater than 0, new state is worse so accept it with some probability
            if score_diff <= 0 or np.random.rand() < np.exp(-score_diff / temperature):
                self.barriers = barriers_prime
                F = F_prime
            # If spread of new state is less than best spread, update the best barrier and spread   
            if F_prime < best_F:
                best_b, best_F = barriers_prime, F_prime
            # Update temperature (temperature *= cooling_rate)
            temperature *= cooling_rate 
        # Return the best barrier placement and fire spread
        print("Best barrier placement: ", best_b)
        print("Best fire spread: ", best_F)
        return best_b
    
    
    # Input state space into this function because we are calling barriers in environment custom now 
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
        fire_env = self.world
        #fire_env = self.env
        # Get the state space of the environment
        state_space = fire_env.get_state()
        (_, rows, cols) = state_space.shape
        paths = self.paths
        populated = self.populated_areas
        flat_paths = np.vstack([np.array(p) for p in paths])

        # Cells not valid for barriers, populated and path cells (Therefore barriers are on grass cells)
        invalid_cells = np.vstack((populated, flat_paths))
    
        # Randomly choose barrier states
        while len(self.barriers) < self.num_barriers:
            # Randomly choose a barrier state
            # REPLACE THIS WITH SIMULATEAD ANNEALING 
            barrier = tuple(np.random.randint(0, [cols, rows])) # Should produce a cell (x, y) in the environment
            # Check if the barrier is not in the populated areas
            if (not np.any(np.all(barrier == invalid_cells, axis=1))) and (barrier not in self.barriers):
                # Set the fuel index to 0 for that cell
                #print("Fuel index before: ", state_space[1][barrier[1]][barrier[0]])
                state_space[1][barrier[0]][barrier[1]] = 0  # 1 = FUEL_INDEX
                #print("Fuel index after: ", state_space[1][barrier[1]][barrier[0]])

                # Add the barrier to the set of barriers
                self.barriers.add(barrier) 
        return self.barriers
        