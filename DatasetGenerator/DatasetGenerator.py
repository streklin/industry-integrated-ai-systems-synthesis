import os
import sys
import random
import pygame

# Ensure the parent directory and Pygame directory are in sys.path so we can import WildFireCA and visualizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Pygame")))

from WildFireCA.WildFireCA import WildfireCA, WildFireState
from visualizer import WildfireVisualizer


class DatasetGenerator:
    """
    Automates generating image datasets of wildfire propagation simulations.
    Generates paired screenshots: base image (Step 0) and ground truth image (Step N).
    """
    def __init__(self, k: int = 1, n: int = 20, m: int = 20, data_dir: str = "data", width: int = 100, height: int = 100, cell_size: int = 8):
        """
        Initialize the Dataset Generator.
        
        Args:
            k: Number of runs/simulations to perform.
            n: Number of steps to simulate for each run before capturing the ground truth.
            m: Upper bound for random simulation steps to run before capturing the base image (default: 20).
            data_dir: Path to directory where screenshots will be saved.
            width: Width of the cellular automata grid.
            height: Height of the cellular automata grid.
            cell_size: Pixel size of each cell in visualizer.
        """
        self.k = k
        self.n = n
        self.m = m
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.cell_size = cell_size

    def _find_flammable_starting_cell(self, ca: WildfireCA):
        """
        Finds a random flammable cell in the grid to start the fire.
        """
        width, height = ca.width, ca.height
        attempts = 0
        while attempts < 1000:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            cell = ca.grid[x][y]
            if cell.state in [WildFireState.GRASSLAND, WildFireState.SHRUB, WildFireState.TREE, WildFireState.HOUSING]:
                return x, y
            attempts += 1
        # Fallback: center cell
        return width // 2, height // 2

    def generate(self):
        """
        Generates the dataset. Runs K simulations, capturing screenshots for the base
        state and the state after N steps.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Starting dataset generation. Generating {self.k} simulation pairs (M={self.m}, N={self.n} steps)...")
        
        # Initialize a single visualizer instance and reuse it across runs to prevent window flickering
        dummy_ca = WildfireCA(width=self.width, height=self.height)
        visualizer = WildfireVisualizer(ca=dummy_ca, cell_size=self.cell_size, window_title="Dataset Generator")

        generated_count = 0
        while generated_count < self.k:
            seed = random.randint(0, 1000000)
            
            # 1. Initialize a new simulation instance
            ca = WildfireCA(width=self.width, height=self.height)
            ca.regenerate(seed=seed)
            
            # 2. Pick a random starting point and ignite
            start_x, start_y = self._find_flammable_starting_cell(ca)
            ca.grid[start_x][start_y].ignite()
            
            # 3. Associate visualizer with the current CA grid
            visualizer.ca = ca
            visualizer.human_agents = []
            
            # 4. Simulate for a random number of steps in [0, M] before capturing the base state
            m_steps = random.randint(0, self.m)
            for step in range(m_steps):
                ca.update()
                pygame.event.pump()
            
            # Check if there are active burning cells at the base state
            burning_cells = sum(
                1 for x in range(self.width) for y in range(self.height)
                if ca.grid[x][y].state == WildFireState.BURNING
            )
            if burning_cells == 0:
                print(f"Skipping Seed: {seed} (M_start: {m_steps}) - fire burned out before capturing base image.")
                continue

            print(f"Running simulation {generated_count+1}/{self.k} (Seed: {seed}, M_start: {m_steps})...")

            # 5. Capture base state (Step m_steps)
            visualizer.refresh()
            base_filename = os.path.join(self.data_dir, f"base/sim_{generated_count:04d}_base.png")
            visualizer.save_screenshot(base_filename)
            
            # 6. Simulate for another N steps as fast as the CPU allows
            for step in range(self.n):
                ca.update()
                pygame.event.pump()
                
            # 7. Capture ground truth state (Step m_steps + N)
            visualizer.refresh()
            gt_filename = os.path.join(self.data_dir, f"gt/sim_{generated_count:04d}_gt.png")
            visualizer.save_screenshot(gt_filename)
            
            generated_count += 1
            
        pygame.quit()
        print(f"\nDataset generation complete. Saved {self.k} pairs to '{self.data_dir}'.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wildfire CA Dataset Generator Utility")
    parser.add_argument("-k", "--k", type=int, default=1, help="Number of simulations to run (default: 1)")
    parser.add_argument("-n", "--n", type=int, default=20, help="Number of steps to simulate (default: 20)")
    parser.add_argument("-m", "--m", type=int, default=20, help="Upper bound for random steps before base state (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save screenshots (default: 'data')")
    parser.add_argument("--width", type=int, default=100, help="Grid width (default: 100)")
    parser.add_argument("--height", type=int, default=100, help="Grid height (default: 100)")
    parser.add_argument("--cell_size", type=int, default=8, help="Pixel size per CA cell (default: 8)")

    args = parser.parse_args()

    # Seed Python's random number generator if a seed is provided to ensure reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        # Seed pygame random if any, though random is used for CA and agent decisions
        print(f"Global seed initialized to: {args.seed}")

    generator = DatasetGenerator(
        k=args.k,
        n=args.n,
        m=args.m,
        data_dir=args.data_dir,
        width=args.width,
        height=args.height,
        cell_size=args.cell_size
    )
    generator.generate()