"""
Example of how to use the wildfire evacuation RL environment.
"""

import gymnasium
import numpy as np
import pyrorl
from src.pyrorl_custom.pyro_custom import WildfireEvacuationEnv


if __name__ == "__main__":

    
    """
    Run basic environment.
    """

    # Set up the environment
    # Set up parameters
    num_rows, num_cols = 10, 10
    populated_areas = np.array([[1, 2], [4, 8], [6, 4], [8, 7]])
    paths = np.array(
        [
            [[1, 0], [1, 1]],
            [[2, 2], [3, 2], [4, 2], [4, 1], [4, 0]],
            [[2, 9], [2, 8], [3, 8]],
            [[5, 8], [6, 8], [6, 9]],
            [[7, 7], [6, 7], [6, 8], [6, 9]],
            [[8, 6], [8, 5], [9, 5]],
            [[8, 5], [9, 5], [7, 5], [7, 4]],
        ],
        dtype=object,
    )
    paths_to_pops = {
        0: [[1, 2]],
        1: [[1, 2]],
        2: [[4, 8]],
        3: [[4, 8]],
        4: [[8, 7]],
        5: [[8, 7]],
        6: [[6, 4]],
    }
    
    env = WildfireEvacuationEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        # Optional parameters you can add:
        # custom_fire_locations=None,  # If you want to specify initial fire locations
        # wind_speed=None,  # Wind speed affecting fire spread
        # wind_angle=None,  # Wind direction angle
        # fuel_mean=8.5,  # Default value for fuel mean
        # fuel_stdev=3,  # Default value for fuel standard deviation
        # fire_propagation_rate=0.094,  # Default value for fire spread rate
        # skip=False  # Whether to skip rendering animation
    )
       
    # Run a simple loop of the environment
    env.reset()

    for _ in range(10):

        # Take action and observation
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        #print("observation: ", observation[1])
        #print("observation shape: ", observation[1].shape)
        total_burned_area = env.burned_area()
        print("Current burned area: ", total_burned_area, "\n")
        # Render environment and print reward
        env.render()
        print("Reward: " + str(reward))
    
    total_burned_area = env.burned_area()
    print("Final area burned: ", total_burned_area, "\n")
    # Generate the gif
    env.unwrapped.generate_gif()