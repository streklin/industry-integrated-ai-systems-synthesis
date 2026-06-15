import os
import sys
import random
import math
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier

# Ensure the parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from WildFireCA.WildFireCA import WildfireCA, WildFireState
from HumanAgents.HumanAgent import HumanAgent, HumanAgentState
from UAVAgents.UAVBase import ReconUAV, FireControlUAV, RescueUAV, UAVState, UAVType

class RLTrainingPipeline:
    def __init__(self, width=100, height=100, num_humans=15, gamma=0.95, init_epsilon=1.0, min_epsilon=0.1, decay_rate=0.7):
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Models directory
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        os.makedirs(self.models_dir, exist_ok=True)

    def _initialize_simulation(self):
        """
        Set up a CA environment and spawn humans.
        """
        seed = random.randint(0, 1000000)
        ca = WildfireCA(width=self.width, height=self.height, seed=seed)
        ca.regenerate(seed=seed)
        
        # Spawn humans
        human_agents = []
        attempts = 0
        while len(human_agents) < self.num_humans and attempts < 1000:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            cell = ca.grid[x][y]
            if cell.state not in [WildFireState.WATER, WildFireState.BURNING, WildFireState.FIRE]:
                activity = random.choice([HumanAgentState.HIKING, HumanAgentState.CAMPING])
                agent = HumanAgent(x=x, y=y, max_x=self.width - 1, max_y=self.height - 1, activity_type=activity)
                human_agents.append(agent)
            attempts += 1
            
        # Ignite a fire
        burnable_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if ca.grid[x][y].state in [WildFireState.GRASSLAND, WildFireState.SHRUB, WildFireState.TREE, WildFireState.HOUSING]:
                    burnable_cells.append(ca.grid[x][y])
        if burnable_cells:
            random.choice(burnable_cells).ignite()
        else:
            ca.grid[self.width // 2][self.height // 2].ignite()
            
        return ca, human_agents

    def collect_expert_demonstrations(self, num_episodes=15):
        """
        Use the heuristic fallback policy to collect initial high-quality trajectories.
        This provides a starting policy classifier far better than random choices.
        """
        print(f"Collecting expert demonstrations ({num_episodes} episodes)...")
        data = {
            UAVType.RECON: {"states": [], "actions": []},
            UAVType.EXTINGUISH: {"states": [], "actions": []},
            UAVType.RESCUE: {"states": [], "actions": []}
        }
        
        for _ in range(num_episodes):
            ca, humans = self._initialize_simulation()
            
            # Spawn agents directly at target cruising areas to ensure high-quality local updates
            agents = [
                ReconUAV(width=self.width, height=self.height),
                FireControlUAV(width=self.width, height=self.height),
                RescueUAV(width=self.width, height=self.height)
            ]
            
            # Place them near the center or near active fire/humans
            for agent in agents:
                agent.x = float(random.randint(40, 60))
                agent.y = float(random.randint(40, 60))
                agent.state = UAVState.CRUISING
                agent.fuel = 100.0
                
            for step in range(50): # limit rollout length
                # Update CA grid every 10 steps
                if step % 10 == 0:
                    ca.update()
                for h in humans:
                    h.update()
                    if ca.grid[h.x][h.y].state == WildFireState.BURNING:
                        h.mark_casualty()
                
                for agent in agents:
                    if agent.fuel <= 0:
                        continue
                    
                    # Get state features
                    state_feat = agent.get_grid_crop_features(ca.grid, humans)
                    # Get heuristic action
                    action = agent.get_heuristic_action(ca.grid, humans)
                    
                    # Store demonstration
                    data[agent.uav_type]["states"].append(state_feat)
                    data[agent.uav_type]["actions"].append(action)
                    
                    # Perform update
                    agent.apply_rl_action(action)
                    agent.update(1.0, ca.grid, humans)
                    
        return data

    def run_rollouts(self, models, num_episodes=20):
        """
        Execute simulation rollouts using current models under epsilon-exploration.
        Computes rewards and stores state action trajectories.
        """
        trajectories = []
        
        for _ in range(num_episodes):
            ca, humans = self._initialize_simulation()
            
            agents = [
                ReconUAV(width=self.width, height=self.height),
                FireControlUAV(width=self.width, height=self.height),
                RescueUAV(width=self.width, height=self.height)
            ]
            
            for agent in agents:
                agent.x = float(random.randint(35, 65))
                agent.y = float(random.randint(35, 65))
                agent.state = UAVState.CRUISING
                agent.fuel = 100.0
                agent.exploration_rate = self.epsilon
                if models and agent.uav_type in models:
                    agent.set_svm_model(models[agent.uav_type])
                    
            episode_history = []
            
            for step in range(100):
                if step % 10 == 0:
                    ca.update()
                for h in humans:
                    h.update()
                    if ca.grid[h.x][h.y].state == WildFireState.BURNING:
                        h.mark_casualty()
                
                step_active = False
                for agent in agents:
                    if agent.fuel <= 0:
                        continue
                    step_active = True
                    
                    # Pre-update values to compute reward impact
                    before_cells = sum(1 for c in agent._get_detected_cells(ca.grid) if ca.grid[c[0]][c[1]].state == WildFireState.BURNING)
                    before_humans_rescued = sum(1 for h in humans if h.activity_type == HumanAgentState.RESCUED)
                    
                    state_feat = agent.get_grid_crop_features(ca.grid, humans)
                    action = agent.select_rl_action(ca.grid, humans)
                    
                    # Update
                    agent.apply_rl_action(action)
                    agent.update(1.0, ca.grid, humans)
                    
                    after_cells = sum(1 for c in agent._get_detected_cells(ca.grid) if ca.grid[c[0]][c[1]].state == WildFireState.BURNING)
                    after_humans_rescued = sum(1 for h in humans if h.activity_type == HumanAgentState.RESCUED)
                    
                    # Compute Reward
                    reward = -0.05 # step penalty
                    
                    if agent.uav_type == UAVType.RECON:
                        # Reward keeping fire and humans in detection range
                        reward += 1.0 * after_cells + 2.0 * len(agent._get_detected_humans(humans))
                    elif agent.uav_type == UAVType.EXTINGUISH:
                        # Extinguished count
                        extinguished = max(0, before_cells - after_cells)
                        reward += 25.0 * extinguished - 1.0 * after_cells # penalize it for leaving fire around
                    elif agent.uav_type == UAVType.RESCUE:
                        # Rescued count
                        rescued = max(0, after_humans_rescued - before_humans_rescued)
                        reward += 50.0 * rescued + 5.0 * len(agent._get_detected_humans(humans))
                        
                    episode_history.append({
                        "uav_type": agent.uav_type,
                        "state": state_feat,
                        "action": action,
                        "reward": reward
                    })
                    
                if not step_active:
                    break
                    
            trajectories.append(episode_history)
            
        return trajectories

    def train_round(self, trajectories, models):
        """
        Train SVR model to estimate action Q-values, determine argmax actions,
        and fit the SVC Classifier.
        """
        # Parse trajectories by agent type
        agent_data = {
            UAVType.RECON: {"states": [], "actions": [], "returns": []},
            UAVType.EXTINGUISH: {"states": [], "actions": [], "returns": []},
            UAVType.RESCUE: {"states": [], "actions": [], "returns": []}
        }
        
        for ep in trajectories:
            # Calculate Monte-Carlo discounted returns
            discounted_returns = []
            cumulative = {UAVType.RECON: 0.0, UAVType.EXTINGUISH: 0.0, UAVType.RESCUE: 0.0}
            
            # Walk backwards to compute returns
            for step in reversed(ep):
                u_type = step["uav_type"]
                cumulative[u_type] = step["reward"] + self.gamma * cumulative[u_type]
                discounted_returns.insert(0, (u_type, step["state"], step["action"], cumulative[u_type]))
                
            for u_type, s, a, ret in discounted_returns:
                agent_data[u_type]["states"].append(s)
                agent_data[u_type]["actions"].append(a)
                agent_data[u_type]["returns"].append(ret)

        updated_models = {}
        for u_type in [UAVType.RECON, UAVType.EXTINGUISH, UAVType.RESCUE]:
            states = np.array(agent_data[u_type]["states"])
            actions = np.array(agent_data[u_type]["actions"])
            returns = np.array(agent_data[u_type]["returns"])
            
            if len(states) == 0:
                # Keep old model if no new data
                if models and u_type in models:
                    updated_models[u_type] = models[u_type]
                continue
                
            print(f"Training policy for {u_type.name} on {len(states)} state transitions...")
            
            # Combine state features and actions to train a Q-value SVR approximator
            # We use a fast Random Forest Regressor here to predict Q-values since SVR fitting is O(N^2)
            # and Random Forest handles mixed categorical/continuous spaces excellently.
            q_inputs = np.column_stack([states, actions])
            q_estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            q_estimator.fit(q_inputs, returns)
            
            # Generate optimal labels for classification
            # For each visited state, evaluate Q-values for actions [0, 1, 2] and pick the best action
            best_actions = []
            for s in states:
                q_vals = []
                for a in [0, 1, 2]:
                    q_vals.append(q_estimator.predict([np.append(s, a)])[0])
                best_actions.append(np.argmax(q_vals))
                
            best_actions = np.array(best_actions)
            
            # Fit SVM Classifier
            if len(np.unique(best_actions)) < 2:
                clf = DummyClassifier(strategy="most_frequent")
            else:
                clf = SVC(kernel='rbf', C=1.0, random_state=42)
            clf.fit(states, best_actions)
            
            updated_models[u_type] = clf
            
        return updated_models

    def save_models(self, models):
        """
        Save the fitted SVC policies to disk.
        """
        for u_type, clf in models.items():
            name = f"{u_type.name.lower()}_policy.pkl"
            filepath = os.path.join(self.models_dir, name)
            with open(filepath, "wb") as f:
                pickle.dump(clf, f)
            print(f"Successfully saved {u_type.name} policy to {filepath}")

    def execute_training(self, num_rounds=5, episodes_per_round=25):
        """
        Run the full training pipeline.
        """
        print("Starting SVM Reinforcement Learning Training Pipeline...")
        
        # 1. Warm start with Heuristic Demonstrations
        demo_data = self.collect_expert_demonstrations(num_episodes=20)
        models = {}
        for u_type in [UAVType.RECON, UAVType.EXTINGUISH, UAVType.RESCUE]:
            states = np.array(demo_data[u_type]["states"])
            actions = np.array(demo_data[u_type]["actions"])
            if len(states) > 0:
                print(f"Initializing {u_type.name} SVC policy with {len(states)} demonstrations...")
                if len(np.unique(actions)) < 2:
                    clf = DummyClassifier(strategy="most_frequent")
                else:
                    clf = SVC(kernel='rbf', C=1.0, random_state=42)
                clf.fit(states, actions)
                models[u_type] = clf

        # 2. Iterate rounds of rollouts, return estimates, and SVM updates
        for r in range(1, num_rounds + 1):
            print(f"\n--- Training Round {r}/{num_rounds} (Epsilon={self.epsilon:.3f}) ---")
            
            # Collect trajectories
            trajectories = self.run_rollouts(models, num_episodes=episodes_per_round)
            
            # Update models
            models = self.train_round(trajectories, models)
            
            # Save progress
            self.save_models(models)
            
            # Decay Epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            
        print("\nTraining completed successfully! All models saved to standard models/ directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SVM RL Agent Trainer")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per training round")
    args = parser.parse_args()
    
    pipeline = RLTrainingPipeline()
    pipeline.execute_training(num_rounds=args.rounds, episodes_per_round=args.episodes)
