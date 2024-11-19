
# import pybullet as p
# import pybullet_data
# import time

# # Connect to PyBullet
# p.connect(p.GUI)  # Use p.DIRECT for headless mode
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load the Plane and Mini Pupper URDF
# p.loadURDF("plane.urdf")
# robot_id = p.loadURDF("/home/vimarsh/Desktop/ROS-Stuff/QuadrupedSim/QuadrupedSim/PyBullet_RL/pupper_env/urdf/pupper.urdf", basePosition=[0, 0, 0.2])

# # Set simulation parameters
# p.setGravity(0, 0, -9.8)
# p.setRealTimeSimulation(1)

# # Visualize Mini Pupper
# while True:
#     p.stepSimulation()
#     time.sleep(0.01)


# # Get joint info
# num_joints = p.getNumJoints(robot_id)
# print(f"Number of Joints: {num_joints}")

# # Set initial joint positions
# for joint in range(num_joints):
#     p.setJointMotorControl2(
#         robot_id,
#         joint,
#         p.POSITION_CONTROL,
#         targetPosition=0,
#         force=5
#     )

# # Test joint movements
# for _ in range(500):
#     for joint in range(num_joints):
#         p.setJointMotorControl2(
#             robot_id,
#             joint,
#             p.POSITION_CONTROL,
#             targetPosition=0.5 * (-1 if joint % 2 == 0 else 1),
#             force=10
#         )
#     p.stepSimulation()
#     time.sleep(0.01)



import gym
from gym import spaces
import numpy as np

class MiniPupperEnv(gym.Env):
    def __init__(self):
        super(MiniPupperEnv, self).__init__()
        self.robot_id = None
        self.init_sim()

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,))

    def init_sim(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        create_rough_terrain()
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("/home/vimarsh/Desktop/ROS-Stuff/QuadrupedSim/QuadrupedSim/PyBullet_RL/pupper_env/urdf/pupper.urdf", basePosition=[0, 0, 0.2])

    def reset(self):
        self.init_sim()
        return self.get_observation()

    def get_observation(self):
        joint_states = p.getJointStates(self.robot_id, range(12))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(joint_positions + joint_velocities + list(base_position) + list(base_orientation))

    def step(self, action):
        for joint, act in enumerate(action):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.TORQUE_CONTROL,
                force=act * 5
            )
        p.stepSimulation()

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.check_done()
        return obs, reward, done, {}

    def compute_reward(self):
        # Reward based on task, e.g., forward movement
        base_velocity = p.getBaseVelocity(self.robot_id)[0]
        return base_velocity[0]  # Encourage forward motion

    def check_done(self):
        base_position, _ = p.getBasePositionAndOrientation(self.robot_id)
        if base_position[2] < 0.1:  # If the robot falls
            return True
        return False

    def render(self, mode='human'):
        pass


import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = MiniPupperEnv()
vec_env = make_vec_env(lambda: env, n_envs=4)

# Define the PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_mini_pupper")

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

def create_rough_terrain():
    for i in range(-10, 10):
        for j in range(-10, 10):
            height = np.random.uniform(0, 0.2)
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, height])
            p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX), basePosition=[i, j, height])

def compute_reward(self):
    base_orientation = p.getBaseOrientation(self.robot_id)
    up_vector = np.array([0, 0, 1])
    reward = np.dot(base_orientation, up_vector)  # Align with vertical axis
    return reward
