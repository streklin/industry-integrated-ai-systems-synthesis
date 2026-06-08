import os
import sys
import random
import pygame

# Ensure the parent directory is in sys.path so we can import WildFireCA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Ensure Pygame directory is also in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from WildFireCA.WildFireCA import WildfireCA, State
from visualizer import WildfireVisualizer

def main():
    # Simulation Parameters
    width, height = 100, 100
    seed = 42
    
    print(f"Initializing Wildfire CA simulation on {width}x{height} grid with seed={seed}...")
    ca = WildfireCA(width=width, height=height, seed=seed)
    ca.regenerate(seed=seed)
    
    # Ignite a starting fire in the middle of the grid
    center_x = width // 2
    center_y = height // 2
    
    # Search nearby to find a flammable cell (Grass, Shrub, or Tree) to ignite
    ignited = False
    for r in range(10):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                tx = center_x + dx
                ty = center_y + dy
                if 0 <= tx < width and 0 <= ty < height:
                    cell = ca.grid[tx][ty]
                    if cell.state in [State.GRASSLAND, State.SHRUB, State.TREE]:
                        cell.ignite()
                        print(f"Starting fire ignited at ({tx}, {ty}) state={cell.state.name}")
                        ignited = True
                        break
            if ignited:
                break
        if ignited:
            break

    if not ignited:
        # Fallback: force ignite the center cell
        ca.grid[center_x][center_y].ignite()
        print(f"Forced start fire ignited at center ({center_x}, {center_y})")

    # Initialize Pygame Visualizer
    # Cell size of 8 pixels gives an 800x800 window
    visualizer = WildfireVisualizer(ca, cell_size=8, window_title="Wildfire CA Simulation")

    # Initial draw before starting simulation updates
    visualizer.refresh()
    
    running = True
    step = 0
    
    print("Starting simulation loop. Press ESC or close the window to quit.")
    
    while running:
        # Check if there are any active burning cells
        burning_cells = sum(
            1 for x in range(width) for y in range(height) 
            if ca.grid[x][y].state == State.BURNING
        )
        
        # Update CA simulation by a single step
        ca.update()
        step += 1
        
        # Refresh the pygame window
        visualizer.refresh()
        
        # Show progress in the console
        if step % 10 == 0 or burning_cells == 0:
            print(f"Step {step}: {burning_cells} cells currently burning.")
            
        # If fire has burned out, pause slightly then wait
        if burning_cells == 0:
            print("The fire has completely burned out. Press ESC or close window to exit.")
            # Keep window open so the user can inspect the final result
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            pygame.quit()
                            sys.exit()
                pygame.time.delay(100)

        # Control simulation speed (approx 10 steps per second)
        visualizer.clock.tick(10)

if __name__ == "__main__":
    main()
