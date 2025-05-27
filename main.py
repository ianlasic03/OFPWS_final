"""
Example of how to use the wildfire evacuation RL environment.
"""

import gymnasium
import numpy as np
import pyrorl
import random
import torch
from src.pyrorl_custom.pyro_custom import WildfireEvacuationEnv
from src.environment.barriers import Barriers


if __name__ == "__main__":
    # Run SA over several seeds
    seeds = [42, 123, 456, 789, 101]  # Different seeds to test
    best_sa_result = float('inf')
    best_sa_barriers = None
    
    # Setup environment 
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
    
    sample_env = WildfireEvacuationEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        barriers=set(),
    )
    # Set up barriers with simulated annealing 
    barrier = Barriers(
        sample_env,
        paths=paths,
        populated_areas=populated_areas,
        num_barriers=5,
    )

    # Fixed action sequence for deterministic evaluation, Do i need this?
    action_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # 10 actions
    
    print("\nRunning SA on multiple seeds:")
    for seed in seeds:
        print(f"\nRunning SA with seed {seed}")
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
    
        # Run SA (Adjust hyper parameters)
        # Get one SA solution per seed
        SA_barriers = barrier.simulated_annealing(
            temperature=1.0,
            cooling_rate=0.999,
            kmax=500,
        )

        # Evaluate the solution
        env = WildfireEvacuationEnv(
            num_rows=num_rows,
            num_cols=num_cols,
            populated_areas=populated_areas,
            paths=paths,
            paths_to_pops=paths_to_pops,
            barriers=SA_barriers
        )
        
        # Run multiple trials to get average performance, for each seed
        trial_results = []
        for _ in range(10):
            env.reset(seed=seed)
            env.barriers = SA_barriers
            env.apply_barriers()
            
            # Use fixed action sequence
            for action in action_sequence:
                observation, reward, terminated, truncated, info = env.step(action)
            
            trial_results.append(env.burned_area())
        
        avg_performance = sum(trial_results) / len(trial_results)
        print(f"Seed {seed} average performance: {avg_performance:.2f}")
        print(f"Individual trial results: {trial_results}")

        # Keep track of best solution
        if avg_performance < best_sa_result:
            best_sa_result = avg_performance
            best_sa_barriers = SA_barriers
    
    print(f"\nBest SA solution found (average across trials): {best_sa_result:.2f}")
    
    # Now evaluate the best solution on multiple seeds
    print("\nEvaluating best solution on multiple seeds:")
    seed_results = []
    
    for seed in seeds:
        print(f"\nRunning with seed {seed}")
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Run multiple trials for this seed
        trial_results = []
        for trial in range(10):
            env.reset(seed=seed)
            # Runninng this barrier solution 
            env.barriers = best_sa_barriers
            env.apply_barriers()
            
            # Use fixed action sequence
            for action in action_sequence:
                observation, reward, terminated, truncated, info = env.step(action)
            
            trial_results.append(env.burned_area())
        
        # Calculate average for this seed
        seed_avg = sum(trial_results) / len(trial_results)
        seed_results.append(seed_avg)
        print(f"Seed {seed} average burned area: {seed_avg:.2f}")
        print(f"Individual trial results: {trial_results}")
    
    # Print overall results
    print("\nOverall Results:")
    print(f"Average across all seeds: {sum(seed_results) / len(seed_results):.2f}")
    print(f"Standard deviation: {np.std(seed_results):.2f}")
    print(f"Min: {min(seed_results):.2f}, Max: {max(seed_results):.2f}")
    
    # Generate the gif from the last trial
    #env.unwrapped.generate_gif()