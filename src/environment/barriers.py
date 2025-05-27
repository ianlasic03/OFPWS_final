import gymnasium
import numpy as np
import pyrorl
import random
import torch
SEED = 42


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

    def exploration(self, point):
        new_point = list(point)
        F_prev = self.env.objective({tuple(new_point)})
        for i in range(2):
            for direction in [-1, 1]:
                test_coord = new_point[i] + direction

                # Boundary checks
                if test_coord < 0:
                    continue
                if i == 0 and test_coord >= self.env.num_rows:
                    continue
                if i == 1 and test_coord >= self.env.num_cols:
                    continue

                test_point = new_point.copy()
                test_point[i] = test_coord
                F_prime = self.env.objective({tuple(test_point)})
                if F_prime < F_prev:
                    new_point = test_point
                    F_prev = F_prime
        return tuple(new_point)

    def hooke_jeeves(self, best_b):
        b_new = set()
        for point in best_b:
            new_point = self.exploration(point)
            if self.env.objective({new_point}) < self.env.objective({point}):
                b_new.add(new_point)
            else:
                b_new.add(point)
        return b_new

    def simulated_annealing(self, HJ, temperature, cooling_rate, kmax):
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
        # Save initial random states
        np_rng_state = np.random.get_state()
        random_rng_state = random.getstate()
        torch_rng_state = torch.get_rng_state()
        
        # Initial random barrier placement
        init_barriers = self.add_barrier()
        wildfire_env = self.env
        F = wildfire_env.objective(init_barriers)
        best_b, best_F = init_barriers, F
        best_random_states = (np_rng_state, random_rng_state, torch_rng_state)
        
        for k in range(kmax):
            print("Current iteration: ", k)
            barriers_prime = wildfire_env.propose(best_b, HJ)
            
            F_prime = wildfire_env.objective(barriers_prime)

            score_diff = F_prime - best_F

            if score_diff <= 0 or np.random.rand() < np.exp(-score_diff / temperature):
                self.barriers = barriers_prime
                F = F_prime
                
            if F_prime < best_F:
                best_b, best_F = barriers_prime, F_prime
                # Save random states when we find a better solution
                best_random_states = (np.random.get_state(), random.getstate(), torch.get_rng_state())
                
            temperature *= cooling_rate
            
        # Restore the random states that produced the best solution
        np.random.set_state(best_random_states[0])
        random.setstate(best_random_states[1])
        torch.set_rng_state(best_random_states[2])
        
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
        