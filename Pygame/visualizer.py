import os
import sys
import pygame

# Ensure the parent directory is in sys.path so we can import WildFireCA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the clean classes directly from WildFireCA.
from WildFireCA.WildFireCA import State, WildfireCA


class WildfireVisualizer:
    """
    Handles visualizing the current state of a WildfireCA simulation grid using Pygame.
    """
    def __init__(self, ca: WildfireCA, cell_size: int = 8, window_title: str = "Wildfire CA Simulation"):
        """
        Initialize the visualizer.
        
        Args:
            ca: The WildfireCA simulation instance to visualize.
            cell_size: Size in pixels of each grid cell.
            window_title: Window title.
        """
        self.ca = ca
        self.cell_size = cell_size
        self.width = ca.width * cell_size
        self.height = ca.height * cell_size
        
        # Color coding:
        # - Ash is grey, Empty is black.
        # - Trees/Shrubs/Grass various shades of green.
        # - Water Blue, Housing orange, urban white, fire red.
        self.colors = {
            State.EMPTY: (0, 0, 0),             # Black
            State.GRASSLAND: (124, 252, 0),     # Lawn Green (Light)
            State.SHRUB: (34, 139, 34),         # Forest Green (Medium)
            State.TREE: (0, 100, 0),            # Dark Green (Dark)
            State.FIRE: (255, 69, 0),           # Orange-Red
            State.BURNING: (255, 0, 0),         # Red
            State.HOUSING: (255, 140, 0),       # Dark Orange
            State.URBAN: (255, 255, 255),       # White
            State.ASH: (128, 128, 128),         # Grey
            State.WATER: (0, 0, 255)            # Blue
        }
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()

    def refresh(self):
        """
        Updates the pygame visualization window to match the current grid state.
        Handles window events to keep the window responsive.
        """
        # Event pump & loop handling to avoid OS "Not Responding" and window hang
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    pygame.quit()
                    sys.exit()

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Render each cell in the grid
        for x in range(self.ca.width):
            for y in range(self.ca.height):
                cell = self.ca.grid[x][y]
                color = self.colors.get(cell.state, (0, 0, 0))
                
                # Draw the cell
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Update full display
        pygame.display.flip()
