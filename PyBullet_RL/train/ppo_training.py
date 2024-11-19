import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

import sys
import os

# Add the folder containing the module to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from pupper_env.pupper_env import make_env

def train_handstand():
    # Initialize wandb for experiment tracking
    wandb.init(
        project="pupper-handstand",
        config={
            "algorithm": "PPO",
            "env": "PupperHandstand",
            "total_timesteps": 1_000_000
        }
    )

    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_handstand_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Train the model
    model.learn(
        total_timesteps=1_000_000,
        callback=wandb.pytorch.WandbCallback()
    )
    
    # Save the model
    model.save("pupper_handstand_ppo")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    train_handstand()