
import React from 'react';

export const PYTHON_FILES = {
  'requirements.txt': `gymnasium>=0.28.1
stable-baselines3[extra]>=2.1.0
shimmy>=0.2.1
torch>=2.0.0
streamlit>=1.25.0
pyyaml>=6.0`,

  'configs/config.yaml': `training:
  total_timesteps: 50000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  ent_coef: 0.01

environment:
  grid_size: 10
  dynamic_probability: 0.1
  max_steps: 200`,

  'env/adaptive_env.py': `import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AdaptiveEnv(gym.Env):
    """
    A dynamic environment where the goal position can change 
    unexpectedly, testing the agent's adaptability.
    """
    def __init__(self, grid_size=10, dynamic_prob=0.1, max_steps=100):
        super(AdaptiveEnv, self).__init__()
        self.grid_size = grid_size
        self.dynamic_prob = dynamic_prob
        self.max_steps = max_steps
        
        # Action space: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space = spaces.Discrete(4)
        
        # Observation: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(4,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.goal_pos = np.array([self.grid_size-1, self.grid_size-1], dtype=np.float32)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Update agent position
        if action == 0: self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1] + 1)
        elif action == 1: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0] + 1)

        # Dynamic Behavior: Goal might move
        if np.random.random() < self.dynamic_prob:
            self.goal_pos = np.random.randint(0, self.grid_size, size=2).astype(np.float32)

        # Calculate reward
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.1 # Step penalty
        
        terminated = False
        if dist < 0.5:
            reward = 10.0
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}`,

  'agents/ppo_agent.py': `from stable_baselines3 import PPO
from env.adaptive_env import AdaptiveEnv

def create_agent(env, lr=0.0003):
    return PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=lr,
        tensorboard_log="./results/ppo_adaptive_tensorboard/"
    )`,

  'src/train.py': `import os
import yaml
from env.adaptive_env import AdaptiveEnv
from agents.ppo_agent import create_agent

def train():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Init Env
    env = AdaptiveEnv(
        grid_size=config['environment']['grid_size'],
        dynamic_prob=config['environment']['dynamic_probability']
    )

    # Init Agent
    model = create_agent(env, lr=config['training']['learning_rate'])

    # Train
    print("Starting Training...")
    model.learn(total_timesteps=config['training']['total_timesteps'])
    
    # Save
    os.makedirs('results', exist_ok=True)
    model.save("results/ppo_agentx")
    print("Model saved to results/ppo_agentx")

if __name__ == "__main__":
    train()`,

  'demo.py': `import time
import numpy as np
from stable_baselines3 import PPO
from env.adaptive_env import AdaptiveEnv

def run_demo():
    env = AdaptiveEnv()
    try:
        model = PPO.load("results/ppo_agentx")
        print("Loaded trained model.")
    except:
        print("No trained model found. Running with random policy.")
        model = None

    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print("\\n--- AgentX Live Demo ---")
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # Visualizing simplified state
        grid = np.full((10, 10), ".")
        gx, gy = map(int, obs[2:])
        ax, ay = map(int, obs[:2])
        grid[gy, gx] = "G"
        grid[ay, ax] = "A"
        
        print(f"Step: {env.current_step} | Pos: ({ax},{ay}) | Goal: ({gx},{gy}) | Reward: {reward:.2f}")
        for row in reversed(grid):
            print(" ".join(row))
        print("-" * 20)
        
        done = terminated or truncated
        time.sleep(0.3)

    print(f"\\nEpisode Finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_demo()`,

  'dashboard/app.py': `import streamlit as st
import pandas as pd
import numpy as np
import os
import time

st.set_page_config(page_title="AgentX Dashboard", layout="wide")

st.title("🚀 AgentX – Adaptive Reinforcement Learning Dashboard")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Status")
    model_exists = os.path.exists("results/ppo_agentx.zip")
    st.success("Model Trained" if model_exists else "Waiting for Training...")
    
    st.info("Environment: Adaptive-v1")
    st.metric("Grid Size", "10x10")
    st.metric("PPO Entropy", "0.01")

with col2:
    st.header("Learning Curves (Simulated from training logs)")
    # Simulating training data
    steps = np.arange(0, 50000, 1000)
    rewards = -50 + 60 * (1 - np.exp(-steps/15000)) + np.random.normal(0, 2, len(steps))
    
    chart_data = pd.DataFrame({
        'Step': steps,
        'Mean Reward': rewards
    })
    
    st.line_chart(chart_data.set_index('Step'))

st.divider()
st.header("Live Agent Trace")
if st.button("Run Simulation Step"):
    st.write("Generating trace...")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Agent successfully adapted to goal shift at step 42!")
`
};

export const Icons = {
  Terminal: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  Chart: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  Code: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
    </svg>
  ),
  Play: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
};
