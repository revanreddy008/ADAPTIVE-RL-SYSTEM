# AgentX – Adaptive Reinforcement Learning System

## Problem Statement (PS1)
Design and implement an adaptive autonomous agent using Reinforcement Learning that can operate in a dynamic and unpredictable environment, continuously learning optimal behavior without manual reprogramming.

---

## Approach Overview

### Agent
The agent uses Proximal Policy Optimization (PPO), a stable policy-gradient reinforcement learning algorithm that updates policies safely using clipped objective functions.

### Environment
A custom Gymnasium environment simulates a dynamic system where correct decisions depend on changing state values.

### Flow
Environment → State Observation → PPO Policy → Action  
← Reward / Penalty ← Environment Transition  

The agent learns optimal actions through repeated interaction and reward feedback.

---

## Setup Steps

```bash
pip install -r requirements.txt
python demo.py

