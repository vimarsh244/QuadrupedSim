import gymnasium as gym
from stable_baselines3 import PPO
from pupper_env.pupper_env import make_env

def test_handstand():
    # Create environment
    env = make_env()
    
    # Load trained model
    model = PPO.load("pupper_handstand_ppo")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run inference
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    test_handstand()