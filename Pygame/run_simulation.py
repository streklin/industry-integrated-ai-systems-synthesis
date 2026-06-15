import os
import sys
import random
import pygame

# Ensure the parent directory is in sys.path so we can import WildFireCA and HumanAgents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Ensure Pygame directory is also in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from WildFireCA.WildFireCA import WildFireState
from HumanAgents.HumanAgent import HumanAgentState
from visualizer import WildfireVisualizer
from UAVAgents.UAVBase import UAVState
from WildfireUAVEnv import WildfireUAVEnv

def main():
    # Simulation Parameters
    width, height = 100, 100
    seed = random.randint(0, 100000)
    num_humans = 15
    cell_size = 10 # 10 pixels per cell gives a 1000x1000 window
    
    print(f"Initializing Wildfire CA simulation on {width}x{height} grid with seed={seed}...")
    
    # Initialize unified simulation environment
    env = WildfireUAVEnv(
        num_recon_uavs=1,
        num_extinguish_uavs=1,
        width=width,
        height=height,
        num_humans=num_humans,
        seed=seed
    )
    env.reset(seed=seed)

    # Initialize Pygame Visualizer (passing human agents and UAV lists)
    visualizer = WildfireVisualizer(
        env.ca,
        human_agents=env.human_agents,
        uavs=env.uav_simulator.uavs,
        cell_size=cell_size,
        window_title="Wildfire, Humans & UAV Simulation"
    )

    # Initial draw before simulation updates
    visualizer.refresh()
    
    running = True
    step = 0
    
    print("Starting simulation loop. Press ESC or close the window to quit.")
    
    while running:
        # Determine hardcoded action to pass to env
        action = []
        recon_uav = env.uav_simulator.uavs[0]
        extinguish_uav = env.uav_simulator.uavs[1]
        
        if "FIRE" in recon_uav.latest_messages:
            if extinguish_uav.state == UAVState.HANGER:
                fire_x = recon_uav.x
                fire_y = recon_uav.y
                action = ["DEPLOY", "1", str(int(fire_x)), str(int(fire_y)), "TRANSMIT"]
                print(f"[Dispatch] Recon UAV detected fire! Deploying Extinguish UAV to coordinates: ({fire_x:.1f}, {fire_y:.1f})")
        
        # Step the unified simulation environment
        status_messages, reward, done, info = env.step(action)
        step = info["step"]
        burning_cells = info["burning_cells"]
        alive = info["humans_alive"]
        casualties = info["casualties"]
        
        # Refresh the pygame visualization
        visualizer.refresh()
        
        # Show status in console
        if step % 10 == 0 or burning_cells == 0:
            print(f"Step {step:02d} | Burning Cells: {burning_cells:4d} | Humans Alive: {alive:2d} | Casualties: {casualties:2d}")
            
        # If fire has burned out, wait for exit
        if burning_cells == 0:
            print(f"\nFire burned out at step {step}. Final Humans State - Alive: {alive}, Casualties: {casualties}.")
            print("Press ESC or close the window to exit.")
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

        # Control simulation speed dynamically using visualizer.fps
        visualizer.clock.tick(visualizer.fps)

if __name__ == "__main__":
    main()
