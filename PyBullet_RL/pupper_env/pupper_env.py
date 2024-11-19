import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import torch

class PupperHandstandEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Gym interface setup
        self.render_mode = render_mode
        
        # Action and observation spaces
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.n_joints = len(self.joint_indices)
        
        # Action space: Torque control for each joint
        self.action_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_joints * 2,),  # Joint angles + joint velocities
            dtype=np.float32
        )
        
        # PyBullet setup
        self.physics_client = None
        self.robot = None
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize PyBullet
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode == 'human' else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
        
        # Reset simulation
        p.resetSimulation()
        
        # Load plane and robot
        p.loadURDF("plane.urdf")
        urdf_path = "pupper_env/urdf/pupper.urdf"
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.5])
        
        # Initial joint configuration for handstand attempt
        initial_pose = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot, joint_index, initial_pose[i])
        
        # Observation after reset
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        # Apply joint torques
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot, 
                jointIndex=joint_index, 
                controlMode=p.TORQUE_CONTROL, 
                force=action[i] * 50  # Scale torque
            )
        
        # Simulate one step
        p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(observation)
        
        # Check termination
        done = self._check_termination(observation)
        
        info = {}
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        # Get joint states
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        
        # Extract joint angles and velocities
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        return np.concatenate([joint_angles, joint_velocities])
    
    def _compute_reward(self, observation):
        # Reward shaping for handstand
        base_orientation = p.getBasePositionAndOrientation(self.robot)[1]
        
        # Convert orientation to Euler angles
        euler_angles = p.getEulerFromQuaternion(base_orientation)
        
        # Reward components
        upright_reward = -abs(euler_angles[0]) - abs(euler_angles[1])  # Closer to vertical
        stability_reward = -np.std(observation[:self.n_joints])  # Minimize joint angle variation
        
        # Total reward
        total_reward = upright_reward + 0.5 * stability_reward
        return total_reward
    
    def _check_termination(self, observation):
        # Base position
        base_pos = p.getBasePositionAndOrientation(self.robot)[0]
        
        # Termination conditions
        fall_height_threshold = 0.1
        max_episode_steps = 1000
        
        if base_pos[2] < fall_height_threshold:
            return True
        
        return False
    
    def close(self):
        p.disconnect()

def make_env():
    return PupperHandstandEnv(render_mode='human')