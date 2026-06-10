from __future__ import annotations
from enum import Enum
from typing import List
import random

import math
from perlin_noise import PerlinNoise


class WildFireState(Enum):
    """
    Represents the state of a cell in the Wildfire Simualtion Cellular Automata.
    """
    EMPTY = 0
    GRASSLAND = 1
    SHRUB = 2
    TREE = 3
    FIRE = 4
    BURNING = 5
    HOUSING = 6
    URBAN = 7
    ASH = 8
    WATER = 9

class CACell:
    """
    Represents a single cell in the Wildfire Simualtion Cellular Automata.
    Human activity, and fire fighting drones are simulated on separate layers.
    """
    def __init__(self, x, y, fuel=10, windspeed=0, direction=(0,0), base_burn_threshold=0.01, seed=None):
        """
        Initialize the cell.
        """
        self.x = x
        self.y = y
        self.previous_state = WildFireState.EMPTY
        self.state = WildFireState.EMPTY
        self.fuel = fuel
        self.windspeed = windspeed # speed of wind
        self.wind_direction = direction # direction of windspeed
        self.base_burn_threshold = base_burn_threshold

        # will need to find some physical justification for values like this.
        self.veg_shrub_burn_modidifer = 0.3
        self.veg_grassland_burn_modifier = 0.1

        self.k = 1 # scaling paramter for ignition probability

    def _can_update(self) -> bool:
        """
        Check if the cell can be updated. (Not every cell type can change or burn)
        """
        return self.state != WildFireState.EMPTY \
            and self.state != WildFireState.ASH \
            and self.state != WildFireState.WATER \
            and self.state != WildFireState.URBAN

    def _check_for_ignition(self, neighbors:List[CACell]) -> bool:
        """
        Check if the cell ignites.
        """
        fire_dir_x, fire_dir_y = 0.0, 0.0
        num_burning = 0
        
        # count the number of burning neighbours and the direction of the fire
        for neighbor in neighbors:
            if neighbor.state == WildFireState.BURNING:
                num_burning += 1
                dx = self.x - neighbor.x
                dy = self.y - neighbor.y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    fire_dir_x += dx / dist
                    fire_dir_y += dy / dist

        # we won't have spontaneous ignition on this simulation
        if num_burning == 0:
            return False

        # calculate the average wind direction over this nhd.
        avg_wind_x = sum(n.windspeed * n.direction[0] for n in neighbors) / len(neighbors)
        avg_wind_y = sum(n.windspeed * n.direction[1] for n in neighbors) / len(neighbors)

        # project the fire direction vector onto the wind direction vector
        wind_alignment = fire_dir_x * avg_wind_x + fire_dir_y * avg_wind_y

        # calculate probability of ignition
        p_ignite = 1 / (1 + math.exp(-self.k * num_burning - wind_alignment - self.base_burn_threshold))

        # different types of vegatation has different probabilities of lighting on fire
        if self.state == WildFireState.SHRUB:
            p_ignite *= self.veg_shrub_burn_modidifer
        elif self.state == WildFireState.GRASSLAND:
            p_ignite *= self.veg_grassland_burn_modifier

        return random.random() < p_ignite

    def _update_burning(self):
        """
        Update the state of the cell based on the state of its neighbors.
        """
        self.fuel -= 1
        if self.fuel <= 0:
            self.previous_state = self.state
            self.state = WildFireState.ASH

    
    def ignite(self):
        """
        Ignite the cell.
        """
        self.previous_state = self.state
        self.state = WildFireState.BURNING

    def update(self, neighbors: List[CACell]):
        """
        Update the state of the cell based on the state of its neighbors.
        """
        if not self._can_update():
            return

        if self.state != WildFireState.BURNING:
            if self._check_for_ignition(neighbors):
                self.ignite()

        if self.state == WildFireState.BURNING:
            self._update_burning()

    def extinguish(self):
        """
        Returns a burning cell state to it pre-burn state. (Fuel is not replenished)
        """
        if self.state != WildFireState.BURNING:
            return
        self.state = self.previous_state
    
class WildfireCA():
    """
    Implements the Wildfire CA.
    """
    def __init__(self, width, height, seed=None):
        """
        Initialize the Wildfire CA.
        """
        self.width = width
        self.height = height
        self.seed = seed
        self.grid = [[CACell(x, y, seed=seed) for y in range(height)] for x in range(width)]
        self.windspeed = 0
        self.wind_direction = (0,0)

    def _get_neighbors(self, x, y):
        """
        Get the neighbors of a cell.
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.grid[nx][ny])
        return neighbors


    def _generate_terrain(self, seed=None):
        """
        Generate the terrain using Perlin noise.
        """
        if seed is None:
            seed = random.randint(0, 1000000)
        self.seed = seed

        noise = PerlinNoise(octaves=4, seed=seed)

        # Scale is adjusted to 0.05 to allow features to vary across a 100x100 grid.
        # Too small of a scale (e.g. 0.001) results in flat terrain with no variation.
        scale = 0.03

        for y in range(self.height):
            for x in range(self.width):
                val = noise([x * scale, y * scale])

                # Stretch the raw noise range (typically [-0.6, 0.6]) to fit the [0, 1] thresholds.
                val = val * 1.4
                val = (val + 1) / 2
                val = max(0.0, min(1.0, val))  # Clamp to [0, 1]

                if val < 0.3:
                    self.grid[x][y].state = WildFireState.WATER
                elif val < 0.5:
                    self.grid[x][y].state = WildFireState.GRASSLAND
                    self.grid[x][y].fuel = 2
                elif val < 0.65:
                    self.grid[x][y].state = WildFireState.SHRUB
                    self.grid[x][y].fuel = 8
                elif val < 0.85:
                    self.grid[x][y].state = WildFireState.TREE
                    self.grid[x][y].fuel = 20
                elif val < 0.95:
                    self.grid[x][y].state = WildFireState.HOUSING
                    self.grid[x][y].fuel = 40
                else:
                    self.grid[x][y].state = WildFireState.URBAN


    def _generate_wind(self, seed=None):
        """
        Generate wind speed and direction for each cell. To keep things simple, we will assume
        the wind remains static during this simulation.
        """
        noise_speed = PerlinNoise(octaves=3, seed=seed)
        noise_dir = PerlinNoise(octaves=3, seed=seed + 1)

        scale_speed = 0.01 # higher values = choppier wind
        scale_dir = 0.03   # higher values = more chaotic direction

        for y in range(self.height):
            for x in range(self.width):
                speed_raw = noise_speed([x * scale_speed, y * scale_speed])
                self.grid[x][y].windspeed = speed_raw

                angle_raw = noise_dir([x * scale_dir, y * scale_dir])
                self.grid[x][y].direction = (angle_raw, angle_raw)

    def regenerate(self, seed=None):
        """
        Regenerate the terrain and wind.
        """
        if seed is None:
            seed = random.randint(0, 1000000)
        self.seed = seed
        self._generate_terrain(seed)
        self._generate_wind(seed)


    def update(self):
        """
        Update the state of the CA by a single step
        """
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y].update(self._get_neighbors(x, y))
                
    