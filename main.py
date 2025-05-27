"""
Example of how to use the wildfire evacuation RL environment.
"""
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
import numpy as np
import pyrorl
import random
import torch
from src.pyrorl_custom.pyro_custom import WildfireEvacuationEnv
from src.environment.barriers import Barriers
import argparse
import matplotlib.pyplot as plt
import time

def train_dqn_policy(env, total_timesteps=10000):
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_dqn_policy(model, env, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"Mean reward over {num_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

def random_barriers(barrier):
    return barrier.add_barrier()
    
def plot_optimization_history(sa_history, fixed_hj_history, adaptive_hj_history, kmax):
    plt.figure(figsize=(10, 6))
    plt.plot(range(kmax), sa_history, 'b-', label='Simulated Annealing')
    plt.plot(range(kmax), fixed_hj_history, 'r-', label='SA with Fixed Hooke-Jeeves')
    plt.plot(range(kmax), adaptive_hj_history, 'g-', label='SA with Adaptive Hooke-Jeeves')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (Burned Area)')
    plt.title('Optimization History Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('optimization_history_comparison.png')
    plt.close()

def SA_training(env, barrier, seeds, action_sequence, HJ, fixed, adaptive):
    best_sa_result = float('inf')
    best_sa_barriers = None
    best_sa_env = None
    best_obj_history = None
    print("\nRunning SA on multiple seeds:")
    for seed in seeds:
        print(f"\nRunning SA with seed {seed}")
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
    
        # Run SA (Adjust hyper parameters)
        # Get one SA solution per seed
        SA_barriers, current_obj_history = barrier.simulated_annealing(
            HJ,
            fixed,
            adaptive,
            temperature=1.0,
            cooling_rate=0.999,
            kmax=500,
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
            best_obj_history = current_obj_history
            # Create new environment with best barriers
            best_sa_env = WildfireEvacuationEnv(
                num_rows=env.num_rows,
                num_cols=env.num_cols,
                populated_areas=env.populated_areas,
                paths=env.paths,
                paths_to_pops=env.paths_to_pops,
                barriers=best_sa_barriers
            )
    
    print(f"\nBest SA solution found (average across trials): {best_sa_result:.2f}")
    return best_sa_barriers, best_sa_env, best_obj_history

"""def random_baseline_eval(barrier, base_env, seeds, action_sequence, num_random=10):
    performances = []
    best_performance = float('inf')
    best_env = None
    best_barriers = None
    
    for i in range(num_random):
        print(f"\nEvaluating random barrier config {i+1}/{num_random}")
        rand_barriers = random_barriers(barrier)
        print(rand_barriers)
        rand_env = WildfireEvacuationEnv(
            num_rows=base_env.num_rows,
            num_cols=base_env.num_cols,
            populated_areas=base_env.populated_areas,
            paths=base_env.paths,
            paths_to_pops=base_env.paths_to_pops,
            barriers=rand_barriers
        )
        seed_results = evaluation(rand_env, seeds, rand_barriers, action_sequence)
        mean = np.mean(seed_results)
        performances.append(mean)
        print(f"Random Config {i+1} mean burned area: {mean:.2f}")
        
        # Track best solution
        if mean < best_performance:
            best_performance = mean
            best_env = rand_env
            best_barriers = rand_barriers
    
    return  seed_results, performances, best_barriers, best_env"""
    

def evaluation(env, model, seeds, barriers, num_trials=10):
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
        for trial in range(num_trials):
            obs, _ = env.reset(seed=seed)
            # Running this barrier solution 
            env.barriers = barriers
            env.apply_barriers()
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)  # Convert numpy array to integer
                obs, reward, terminated, truncated, info = env.step(action)
            
            trial_results.append(env.burned_area())
        
        # Calculate average for this seed
        seed_avg = np.mean(trial_results)
        seed_results.append(seed_avg)
        print(f"Seed {seed} average burned area: {seed_avg:.2f}")
        print(f"Individual trial results: {trial_results}")
    return seed_results

def render_best_solution(env, seeds, model):
    """Render the best solution for visualization using the best seed"""
    print("\nRendering best solution...")
    
    # Find the best seed by running one trial per seed
    best_seed = None
    best_performance = float('inf')
    
    print("Finding best seed...")
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        env.apply_barriers()
        
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert numpy array to integer
            obs, reward, terminated, truncated, info = env.step(action)
        
        performance = env.burned_area()
        print(f"Seed {seed} performance: {performance:.2f}")
        
        if performance < best_performance:
            best_performance = performance
            best_seed = seed
    
    print(f"\nRendering best solution with best seed {best_seed} (burned area: {best_performance:.2f})")
    
    # Render the best seed
    obs, _ = env.reset(seed=best_seed)
    env.apply_barriers()
    
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  # Convert numpy array to integer
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # This will create screenshots
    
    # Generate gif from the screenshots
    env.generate_gif()

def main():
    parser = argparse.ArgumentParser(description='Run pyrorl experiments')

    parser.add_argument('--random', action='store_true', default=False,
                      help='Random Barriers')
    parser.add_argument('--SA', action='store_true', default=False,
                      help='Simulated annealing Barriers')
    parser.add_argument('--HJ', action='store_true', default=False,
                      help='Use Hooke Jeeves')
    parser.add_argument('--fixed', action='store_true', default=False,
                      help='Use Fixed schedule for Hooke Jeeves')
    parser.add_argument('--adaptive', action='store_true', default=False,
                      help='Use adaptive schedule for Hooke Jeeves')
    parser.add_argument('--render', action='store_true', default=False,
                      help='Render the best solution')
   
    args = parser.parse_args()

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
    
    # Create base environment
    base_env = WildfireEvacuationEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        populated_areas=populated_areas,
        paths=paths,
        paths_to_pops=paths_to_pops,
        barriers=set(),
    )
    
    # Set up barriers manager
    barrier = Barriers(
        base_env,
        paths=paths,
        populated_areas=populated_areas,
        num_barriers=5,
    )

    # Fixed action sequence for deterministic evaluation
    action_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # 10 actions
    
    if args.random:
        random_barrier_set = random_barriers(barrier)
        rand_env = WildfireEvacuationEnv(
            num_rows=num_rows,
            num_cols=num_cols,
            populated_areas=populated_areas,
            paths=paths,
            paths_to_pops=paths_to_pops,
            barriers=random_barrier_set,
        )
        # Train DQN directly on the environment
        dqn_model = train_dqn_policy(rand_env, total_timesteps=20000)
        evaluate_dqn_policy(dqn_model, rand_env)
        seed_results = evaluation(rand_env, dqn_model, seeds, random_barrier_set)
        
        # Print overall results
        print("\nRandom Results:")
        print(f"Average across all seeds: {sum(seed_results) / len(seed_results):.2f}")
        print(f"Standard deviation: {np.std(seed_results):.2f}")
        print(f"Min: {min(seed_results):.2f}, Max: {max(seed_results):.2f}")

        if args.render:
            render_best_solution(rand_env, seeds, dqn_model)
        del dqn_model

    if args.SA: 
        # Run regular SA
        start_time = time.time()
        HJ = False
        fixed = False
        adaptive = False
        best_sa_barriers, best_sa_env, sa_history = SA_training(base_env, barrier, seeds, action_sequence, HJ, fixed, adaptive)
        sa_time = time.time() - start_time
        print(f"\nSA runtime: {sa_time:.2f} seconds")

        # Run SA with Fixed HJ
        start_time = time.time()
        HJ = True
        fixed = True
        adaptive = False
        best_sa_fixed_hj_barriers, best_sa_fixed_hj_env, fixed_hj_history = SA_training(base_env, barrier, seeds, action_sequence, HJ, fixed, adaptive)
        fixed_hj_time = time.time() - start_time
        print(f"\nSA with Fixed Hooke-Jeeves runtime: {fixed_hj_time:.2f} seconds")

        # Run SA with Adaptive HJ
        start_time = time.time()
        HJ = True
        fixed = False
        adaptive = True
        best_sa_adaptive_hj_barriers, best_sa_adaptive_hj_env, adaptive_hj_history = SA_training(base_env, barrier, seeds, action_sequence, HJ, fixed, adaptive)
        adaptive_hj_time = time.time() - start_time
        print(f"\nSA with Adaptive Hooke-Jeeves runtime: {adaptive_hj_time:.2f} seconds")

        # Plot all three histories together
        plot_optimization_history(sa_history, fixed_hj_history, adaptive_hj_history, 500)

        # Find the best solution among all three methods
        final_values = {
            'SA': sa_history[-1],
            'Fixed HJ': fixed_hj_history[-1],
            'Adaptive HJ': adaptive_hj_history[-1]
        }
        best_method = min(final_values.items(), key=lambda x: x[1])[0]
        print(f"\nBest method: {best_method}")

        # Use the best solution for DQN training
        if best_method == 'SA':
            best_sa_env = best_sa_env
            best_sa_barriers = best_sa_barriers
        elif best_method == 'Fixed HJ':
            best_sa_env = best_sa_fixed_hj_env
            best_sa_barriers = best_sa_fixed_hj_barriers
        else:  # Adaptive HJ
            best_sa_env = best_sa_adaptive_hj_env
            best_sa_barriers = best_sa_adaptive_hj_barriers

        # Train DQN directly on the environment
        dqn_model = train_dqn_policy(best_sa_env, total_timesteps=20000)
        evaluate_dqn_policy(dqn_model, best_sa_env)

        # Evaluate best SA barriers
        seed_results = evaluation(best_sa_env, dqn_model, seeds, best_sa_barriers)
        
        # Print overall results
        print("\nSA Results:")
        print(f"Average across all seeds: {sum(seed_results) / len(seed_results):.2f}")
        print(f"Standard deviation: {np.std(seed_results):.2f}")
        print(f"Min: {min(seed_results):.2f}, Max: {max(seed_results):.2f}")
        
        if args.render:
            render_best_solution(best_sa_env, seeds, dqn_model)
        del dqn_model


if __name__ == "__main__":
    main()