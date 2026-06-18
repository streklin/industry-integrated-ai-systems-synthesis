import os
import sys
import random
import pygame
import pickle

# Force UTF-8 output so LLM responses containing Unicode (e.g. → arrows)
# don't crash with UnicodeEncodeError on Windows consoles using cp1252.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure the parent directory is in sys.path so we can import WildFireCA and HumanAgents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Ensure Pygame directory is also in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from WildFireCA.WildFireCA import WildfireCA, WildFireState
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState
from visualizer import WildfireVisualizer
from UAVAgents.UAVBase import ReconUAV, FireControlUAV, RescueUAV, UAVState, UAVType
from UAVAgents.UAVSimulatorModule import UAVSimulator
from simulation_coordinator import SimulationCoordinator

def main():
    # Simulation Parameters
    width, height = 100, 100
    seed = random.randint(0, 100000)
    num_humans = 15
    cell_size = 10 # 10 pixels per cell gives a 1000x1000 window
    
    print(f"Initializing Wildfire CA simulation on {width}x{height} grid with seed={seed}...")
    ca = WildfireCA(width=width, height=height, seed=seed)
    ca.regenerate(seed=seed)
    
    # 1. Spawn human agents on random dry land (not water, not burning)
    human_agents = []
    attempts = 0
    while len(human_agents) < num_humans and attempts < 1000:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        cell = ca.grid[x][y]
        
        if cell.state not in [WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE]:
            # Alternate activity types
            activity = random.choice([HumanAgentState.HIKING, HumanAgentState.CAMPING])
            # Bounding limits should be width-1 and height-1 to match index ranges
            agent = HumanAgent(x=x, y=y, max_x=width - 1, max_y=height - 1, activity_type=activity)
            human_agents.append(agent)
        attempts += 1

    print(f"Successfully spawned {len(human_agents)} human agents in the wilderness.")
    
    # Ignite a starting fire in any burnable cell on the grid randomly
    burnable_cells = []
    for x in range(width):
        for y in range(height):
            cell = ca.grid[x][y]
            if cell.state in [WildFireState.GRASSLAND, WildFireState.SHRUB, WildFireState.TREE, WildFireState.HOUSING]:
                burnable_cells.append(cell)
                
    if burnable_cells:
        selected_cell = random.choice(burnable_cells)
        selected_cell.ignite()
        print(f"Starting fire ignited randomly at ({selected_cell.x}, {selected_cell.y}) state={selected_cell.state.name}")
    else:
        center_x = width // 2
        center_y = height // 2
        ca.grid[center_x][center_y].ignite()
        print(f"Forced start fire ignited at center ({center_x}, {center_y})")

    # Pick a random home base on dry, non-burning land
    safe_cells = [
        (x, y)
        for x in range(width) for y in range(height)
        if ca.grid[x][y].state not in [
            WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE, WildFireState.ASH
        ]
    ]
    base_x, base_y = random.choice(safe_cells)
    base_x, base_y = float(base_x), float(base_y)
    print(f"Home base placed at ({base_x:.0f}, {base_y:.0f})")

    # Initialize UAV agents
    uavs = []
    uav_id_counter = 0

    # 5 Recon UAVs
    for i in range(5):
        uav = ReconUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # 2 Extinguish UAVs
    for i in range(2):
        uav = FireControlUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # 1 Rescue UAV
    for i in range(1):
        uav = RescueUAV(uav_id=uav_id_counter, width=width, height=height)
        uavs.append(uav)
        uav_id_counter += 1

    # Apply shared home base to all UAVs — must be done after construction
    # so that home_base, x, and y are all consistent from the first tick.
    for uav in uavs:
        uav.home_base = (base_x, base_y)
        uav.x = base_x
        uav.y = base_y


    # Initialize Coordinator
    coordinator = SimulationCoordinator(n_steps=40)

    # Load SVM policies from models directory if available
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UAVAgents", "models"))
    policy_files = {
        ReconUAV: "recon_policy.pkl",
        FireControlUAV: "extinguish_policy.pkl",
        RescueUAV: "rescue_policy.pkl"
    }

    for uav in uavs:
        filename = policy_files.get(type(uav))
        if filename:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        model = pickle.load(f)
                    uav.set_svm_model(model)
                    print(f"Loaded SVM policy model for {uav.uav_type.name} (ID: {uav.uav_id}) from {filepath}")
                except Exception as e:
                    print(f"Error loading SVM model for {uav.uav_type.name} (ID: {uav.uav_id}): {e}")

    uav_simulator = UAVSimulator(uavs)

    # Initialize Pygame Visualizer (passing human agents and UAV lists)
    visualizer = WildfireVisualizer(ca, human_agents=human_agents, uavs=uav_simulator.uavs, cell_size=cell_size, window_title="Wildfire, Humans & UAV Simulation", is_recording=True)

    # Initial draw before simulation updates
    visualizer.refresh()
    
    running = True
    step = 0
    _recording_saved = False

    print("Starting simulation loop. Press ESC or close the window to quit.")

    def _save_recording():
        """Save recording once. Subsequent calls are no-ops."""
        nonlocal _recording_saved
        if _recording_saved:
            return
        _recording_saved = True
        if visualizer.is_recording and visualizer.frames:
            print("SAVING RECORDING...")
            visualizer.save_recording("video/human_behavior.mp4")
        elif not visualizer.frames:
            print("[Recording] No frames captured — nothing to save.")

    def _save_and_quit():
        """Save the recording (if enabled) then exit cleanly."""
        print("EXITING...")
        _save_recording()
        pygame.quit()
        sys.exit()

    # maximum number of simulation steps 1500
    max_steps = 750

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _save_and_quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        _save_and_quit()
                    elif event.key == pygame.K_SPACE:
                        print("\n[Simulation Paused]")
                        user_query = input("Enter your query for the Command Center Assistant: ")
                        if user_query.strip():
                            print("Querying Assistant Agent...")
                            response = coordinator.command_center.query(user_query)
                            print(f"\n[Assistant Response]:\n{response}\n")
                        print("[Simulation Resumed]\n")

            if not running:
                break

            # Check active fire count
            burning_cells = sum(
                1 for x in range(width) for y in range(height) 
                if ca.grid[x][y].state == WildFireState.BURNING
            )
            
            # 2. Update human agents logic
            for agent in human_agents:
                agent.update()
                
                # Check if current cell is burning
                if ca.grid[agent.x][agent.y].state == WildFireState.BURNING:
                    agent.mark_casualty()
            
            # 3. Update UAV simulation and Dispatch Decision logic
            uav_simulator.update(delta_time=1.0, ca_grid=ca.grid, humans=human_agents)
            
            # Trigger Coordinator to communicate with CommandCenterAgent
            coordinator.on_step(step, visualizer, uav_simulator.uavs, human_agents)

            # 4. Update CA simulation step (once every 10 steps)
            if step % 10 == 0:
                ca.update()
            step += 1
            
            # Refresh the pygame visualization
            visualizer.refresh()
            
            # Calculate human statistics
            casualties = sum(1 for a in human_agents if a.activity_type == HumanAgentState.CASUALTY)
            rescued = sum(1 for a in human_agents if a.activity_type == HumanAgentState.RESCUED)
            alive = len(human_agents) - casualties - rescued
            
            # Show status in console
            if step % 10 == 0 or burning_cells == 0:
                print(f"Step {step:02d} | Burning Cells: {burning_cells:4d} | Humans Alive: {alive:2d} | Rescued: {rescued:2d} | Casualties: {casualties:2d}")
            
            max_steps -= 1

            # If fire has burned out, wait for user to exit
            if burning_cells == 0 or max_steps == 0:
                print(f"\nFire burned out or simulation timed out at step {step}. Final Humans State - Alive: {alive}, Rescued: {rescued}, Casualties: {casualties}.")
                print("Press ESC or close the window to exit.")
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            _save_and_quit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                _save_and_quit()
                    pygame.time.delay(100)

            # Control simulation speed dynamically using visualizer.fps
            visualizer.clock.tick(visualizer.fps)

    finally:
        # Guaranteed to run no matter how the loop exits: normal exit,
        # exception from coordinator API call, KeyboardInterrupt, etc.
        _save_recording()
        pygame.quit()


if __name__ == "__main__":
    main()

