"""
OpenAI Gym Environment Wrapper Class
"""
from src.environment.environment_custom import FireWorld
#from pyrorl.envs.environment.environment import FireWorld
from src.environment.barriers import Barriers
import gymnasium as gym
from gymnasium import spaces
import imageio.v2 as imageio
import numpy as np
import os
import pygame
import shutil
import sys
import random
from typing import Optional, Any

# Constants for visualization
IMG_DIRECTORY = "grid_screenshots/"
BARRIER_COLOR = pygame.Color("#ffffff")      # white
FIRE_COLOR = pygame.Color("#ef476f")
POPULATED_COLOR = pygame.Color("#073b4c")
EVACUATING_COLOR = pygame.Color("#118ab2")
PATH_COLOR = pygame.Color("#ffd166")
GRASS_COLOR = pygame.Color("#06d6a0")
FINISHED_COLOR = pygame.Color("#BF9ACA")


class WildfireEvacuationEnv(gym.Env):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        populated_areas: np.ndarray,
        paths: np.ndarray,
        paths_to_pops: dict,
        barriers: set,
        custom_fire_locations: Optional[np.ndarray] = None,
        wind_speed: Optional[float] = None,
        wind_angle: Optional[float] = None,
        fuel_mean: float = 8.5,
        fuel_stdev: float = 3,
        fire_propagation_rate: float = 0.094,
        skip: bool = False,
    ):
        """
        Set up the basic environment and its parameters.
        """
        # Save parameters and set up environment
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.populated_areas = populated_areas
        self.paths = paths
        self.paths_to_pops = paths_to_pops
        self.barriers = barriers
        self.custom_fire_locations = custom_fire_locations
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.fuel_mean = fuel_mean
        self.fuel_stdev = fuel_stdev
        self.fire_propagation_rate = fire_propagation_rate
        self.skip = skip
        self.fire_env = FireWorld(
            num_rows,
            num_cols,
            populated_areas,
            paths,
            paths_to_pops,
            custom_fire_locations=custom_fire_locations,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            fuel_mean=fuel_mean,
            fuel_stdev=fuel_stdev,
            fire_propagation_rate=fire_propagation_rate,
        )

        # Apply barriers to the fire_env
        self.apply_barriers()

        # Set up action space
        actions = self.fire_env.get_actions()
        self.action_space = spaces.Discrete(len(actions))

        # Set up observation space
        observations = self.fire_env.get_state()
        self.observation_space = spaces.Box(
            low=0, high=200, shape=observations.shape, dtype=np.float64
        )

        # Create directory to store screenshots
        if os.path.exists(IMG_DIRECTORY) is False:
            os.mkdir(IMG_DIRECTORY)

    def apply_barriers(self):
        """
        Apply barriers to the fire_env by setting the fuel index to 0 for barrier locations.
        """
        for r, c in self.barriers:
            self.fire_env.state_space[1][r][c] = 0  # Set fuel index to 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to its initial state.
        """
        self.fire_env = FireWorld(
            self.num_rows,
            self.num_cols,
            self.populated_areas,
            self.paths,
            self.paths_to_pops,
            wind_speed=self.wind_speed,
            wind_angle=self.wind_angle,
            fuel_mean=self.fuel_mean,
            fuel_stdev=self.fuel_stdev,
            fire_propagation_rate=self.fire_propagation_rate,
        )
        """barrier_manager = Barriers(
            env=self.fire_env, paths=self.paths,
            populated_areas=self.populated_areas,
            num_barriers=5          
        )
        # Run with random placement of barriers
        self.barriers = barrier_manager.add_barrier()"""
        self.apply_barriers()

        state_space = self.fire_env.get_state()
        return state_space, {"": ""}


    def step(self, action: int) -> tuple:
        """
        Take a step and advance the environment after taking an action.
        """
        # Take the action and advance to the next timestep
        self.fire_env.set_action(action)
        # Add barriers as an argument here
        self.fire_env.advance_to_next_timestep(self.barriers)

        # Gather observations and rewards
        observations = self.fire_env.get_state()
        rewards = self.fire_env.get_state_utility()
        terminated = self.fire_env.get_terminated()
        return observations, rewards, terminated, False, {"": ""}

    def render_hf(
        self, screen: pygame.Surface, font: pygame.font.Font
    ) -> pygame.Surface:
        """
        Set up header and footer
        """
        # Get width and height of the screen
        surface_width = screen.get_width()
        surface_height = screen.get_height()

        # Starting locations and timestep
        x_offset, y_offset = 0.05, 0.05
        timestep = self.fire_env.get_timestep()

        # Set title of the screen
        text = font.render("Timestep #: " + str(timestep), True, (0, 0, 0))
        screen.blit(text, (surface_width * x_offset, surface_height * y_offset))

        # Set initial grid squares and offsets
        grid_squares = [
            (BARRIER_COLOR, "Barrier"),
            (GRASS_COLOR, "Grass"),
            (FIRE_COLOR, "Fire"),
            (POPULATED_COLOR, "Populated"),
            (EVACUATING_COLOR, "Evacuating"),
            (PATH_COLOR, "Path"),
            (FINISHED_COLOR, "Finished"),
        ]
        x_offset, y_offset = 0.2, 0.045

        # Iterate through, create the grid squares
        for i in range(len(grid_squares)):

            # Get the color and name, set in the screen
            (color, name) = grid_squares[i]
            pygame.draw.rect(
                screen,
                color,
                (surface_width * x_offset, surface_height * y_offset, 25, 25),
            )
            text = font.render(name, True, (0, 0, 0))
            screen.blit(
                text, (surface_width * x_offset + 35, surface_height * y_offset + 5)
            )

            # Adjust appropriate offset
            x_offset += 0.125

        return screen

    def render(self):
        """
        Render the environment
        """
        # Set up the state space
        state_space = self.fire_env.get_state()
        finished_evacuating = self.fire_env.get_finished_evacuating()
        (_, rows, cols) = state_space.shape
        

        #print("barriers: ", barriers)

        # Get dimensions of the screen
        pygame.init()
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h

        # Set up screen and font
        surface_width = screen_width * 0.8
        surface_height = screen_height * 0.8
        screen = pygame.display.set_mode([surface_width, surface_height])
        font = pygame.font.Font(None, 25)

        # Set screen details
        screen.fill((255, 255, 255))
        pygame.display.set_caption("PyroRL")
        screen = self.render_hf(screen, font)

        # Calculation for square
        total_width = 0.85 * surface_width - 2 * (cols - 1)
        total_height = 0.85 * surface_height - 2 * (rows - 1)
        square_dim = min(int(total_width / cols), int(total_height / rows))

        # Calculate start x, start y
        start_x = surface_width - 2 * (cols - 1) - square_dim * cols
        start_y = (
            surface_height - 2 * (rows - 1) - square_dim * rows + 0.05 * surface_height
        )
        start_x /= 2
        start_y /= 2

        # Running the loop!
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    timestep = self.fire_env.get_timestep()
                    pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                    running = False

            # Iterate through all of the squares
            # Note: try to vectorize?
            for x in range(cols):
                for y in range(rows):
                    # Set color of the square
                    color = GRASS_COLOR
                    if (y, x) in self.barriers:
                        color = BARRIER_COLOR
                    if state_space[4][y][x] > 0:
                        color = PATH_COLOR
                    if state_space[0][y][x] == 1:
                        color = FIRE_COLOR
                    if state_space[2][y][x] == 1:
                        color = POPULATED_COLOR
                    if state_space[3][y][x] > 0:
                        color = EVACUATING_COLOR
                    if [y, x] in finished_evacuating:
                        color = FINISHED_COLOR

                    # Draw the square
                    # self.grid_dim = min(self.grid_width, self.grid_height)
                    square_rect = pygame.Rect(
                        start_x + x * (square_dim + 2),
                        start_y + y * (square_dim + 2),
                        square_dim,
                        square_dim,
                    )
                    pygame.draw.rect(screen, color, square_rect)

            # Render and then quit outside
            pygame.display.flip()

            # If we skip, then we basically just render the canvas and then quit outside
            if self.skip:
                timestep = self.fire_env.get_timestep()
                pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                running = False
        pygame.quit()

    def burned_area(self, fire_env=None):
        """
        Quantify total amount of area burned.
        If a specific fire_env is provided, use it; otherwise, use self.fire_env.
        """
        fire_env = fire_env or self.fire_env
        state_space = fire_env.get_state()
        (_, rows, cols) = state_space.shape
        burned_area = 0
        # Loop through each state
        for x in range(cols):
            for y in range(rows):
                if state_space[0][y][x] == 1:
                    burned_area += 1
        return burned_area
    
    def objective(self, barriers, trials=10):
        results = []
        for _ in range(trials):
            self.reset()
            self.barriers = barriers
            for _ in range(10): # Go through 10 timesteps
                a = self.action_space.sample()
                _, _, done, _, _ = self.step(a)
            results.append(self.burned_area())
        
        avg_result = sum(results) / len(results)
        return avg_result
    
    def exploration(self, point):
        new_point = list(point)
        F_prev = self.objective({tuple(new_point)})
        for i in range(2):
            for direction in [-1, 1]:
                test_coord = new_point[i] + direction

                # Boundary checks
                if test_coord < 0:
                    continue
                if i == 0 and test_coord >= self.num_rows:
                    continue
                if i == 1 and test_coord >= self.num_cols:
                    continue

                test_point = new_point.copy()
                test_point[i] = test_coord
                F_prime = self.objective({tuple(test_point)})
                if F_prime < F_prev:
                    new_point = test_point
                    F_prev = F_prime
        return tuple(new_point)

    def hooke_jeeves(self, B_prime, valid_cells):
        # Use Hooke-Jeeves to find the least effective barrier to remove
        worst_barrier = None
        worst_objective = float('-inf')
        
        # Test removing each barrier
        for barrier in B_prime:
            test_set = B_prime.copy()
            test_set.remove(barrier)
            obj = self.objective(test_set)
            if obj > worst_objective:
                worst_objective = obj
                worst_barrier = barrier
        
        # Remove the worst barrier
        B_prime.remove(worst_barrier)
            
        # Try to find a better point for each remaining barrier
        best_new_point = None
        best_objective = worst_objective
        
        for point in B_prime:
            new_point = self.exploration(point)
            if new_point in valid_cells:
                new_obj = self.objective({new_point})
                if new_obj < best_objective:
                    best_new_point = new_point
                    best_objective = new_obj
            else:
                print(f"Warning: New point {new_point} is not in valid cells. Keeping original point {point}")
        
        # If no better point was found, return the original removed point
        if best_new_point is None:
            print(f"No better point found, keeping original point {worst_barrier}")
            return worst_barrier
            
        return best_new_point
        
    def propose(self, B, HJ):
        B_prime = B.copy()
        paths = self.paths
        populated = self.populated_areas
        flatten_paths = np.vstack([np.array(p) for p in paths])

        # Cells not valid for barriers
        invalid_cells = np.vstack((populated, flatten_paths))
        invalid_cells = np.unique(invalid_cells, axis=0)
        invalid_cells = invalid_cells[(invalid_cells[:, 0] < self.num_rows) & (invalid_cells[:, 1] < self.num_cols)]

        # Compute all valid cells
        all_cells = {(r, c) for r in range(self.num_rows) for c in range(self.num_cols)}
        valid_cells = all_cells - B_prime - set(map(tuple, invalid_cells))

        if not valid_cells:
            raise RuntimeError("No valid cells available for new barriers.")

        if HJ:
            best_new_barrier = self.hooke_jeeves(B_prime, valid_cells)
            best_new_barrier = (np.int64(best_new_barrier[0]), np.int64(best_new_barrier[1]))
            B_prime.add(best_new_barrier)
        else:
            # Random selection for both removal and placement
            i = random.choice(list(B))
            B_prime.remove(i)
            new_barrier = random.choice(list(valid_cells))
            # Turn components of the tuple into np.int64
            new_barrier = (np.int64(new_barrier[0]), np.int64(new_barrier[1]))
            B_prime.add(new_barrier)

        return B_prime


    def generate_gif(self):
        """
        Save run as a GIF.
        """
        files = [str(i) for i in range(1, self.fire_env.get_timestep() + 1)]
        images = [imageio.imread(IMG_DIRECTORY + f + ".png") for f in files]
        imageio.mimsave("training.gif", images, loop=0)
        shutil.rmtree(IMG_DIRECTORY)