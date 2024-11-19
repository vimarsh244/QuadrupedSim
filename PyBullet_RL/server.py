import gym
from gym import spaces
import pybullet as p
import numpy as np
import time

class HyperdogEnv(gym.Env):
    def __init__(self, render=True):
        super(HyperdogEnv, self).__init__()

        # Render flag to toggle between graphical and non-graphical modes
        self.render_flag = render
        
        # Action space: Assume the robot can control 12 actuators (3 per leg, 4 legs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        
        # Observation space: A simple observation could include joint positions and velocities (12 values for joints)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        
        # Initialize PyBullet simulation
        if self.render_flag:
            self.physicsClient = p.connect(p.GUI)  # Use GUI mode for graphical rendering
        else:
            self.physicsClient = p.connect(p.DIRECT)  # Use DIRECT mode for non-graphical simulation
        
        self.robot_id = None
        self.time_step = 0.01  # Time step for each simulation iteration (in seconds)

    def reset(self):
        # Reset the simulation
        p.resetSimulation()
        
        # Load the robot model (replace 'hyperdog.urdf' with your actual robot URDF file)
        self.robot_id = p.loadURDF("hyperdog.urdf", basePosition=[0, 0, 0.5])

        # Set the robot's initial joint positions and velocities (if needed)
        self._set_initial_conditions()

        # Get the initial observation
        observation = self._get_observation()
        
        return observation

    def _set_initial_conditions(self):
        # Example of setting initial conditions: resetting joint positions and velocities
        # Reset joint positions (e.g., zero position for all joints)
        joint_positions = [0.0] * 12  # Example, you can set custom values if needed
        joint_velocities = [0.0] * 12  # Initial joint velocities set to 0

        for i in range(12):
            p.resetJointState(self.robot_id, i, joint_positions[i], joint_velocities[i])

    def step(self, action):
        # Apply the actions to the robot (assume action is a vector of torques or joint positions)
        self._apply_action(action)
        
        # Step the simulation forward
        p.stepSimulation()
        
        # Get the current observation
        observation = self._get_observation()
        
        # Calculate reward (a simple placeholder, you can make this more sophisticated)
        reward = self._calculate_reward()
        
        # Check if the episode is done (for now, we'll assume it ends after 1000 steps)
        done = self._check_done()
        
        return observation, reward, done, {}

    def _apply_action(self, action):
        # Apply the action to each of the robot's joints
        for i in range(12):  # Assume we have 12 joints
            # Action is assumed to be a torque value here, you could map it to other control signals
            p.setJointMotorControl2(self.robot_id, i, p.TORQUE_CONTROL, force=action[i])

    def _get_observation(self):
        # Retrieve joint states (positions and velocities) from the robot
        joint_states = p.getJointStates(self.robot_id, range(12))  # Get states for 12 joints
        
        # Extract positions and velocities
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Combine joint positions and velocities into one observation vector
        observation = np.concatenate([joint_positions, joint_velocities])
        
        return np.array(observation, dtype=np.float32)

    def _calculate_reward(self):
        # Placeholder for reward calculation
        # A simple reward could be based on distance traveled or energy efficiency
        # For now, we'll reward the robot for staying upright
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        reward = -np.linalg.norm(base_pos)  # Penalize if the robot falls away from the center
        
        return reward

    def _check_done(self):
        # Check if the robot has fallen over or exceeded step limit
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        if base_pos[2] < 0.1:  # Robot is considered "fallen" if it is below a certain height
            return True
        return False

    def render(self, mode='human'):
        # Render the environment (only needed if graphical rendering is required)
        if self.render_flag:
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

    def close(self):
        # Close the environment
        p.disconnect()

def test_hyperdog_env():
    # Create the environment
    env = HyperdogEnv(render=True)  # Set render=True for graphical rendering
    
    # Reset the environment
    observation = env.reset()
    print("Initial Observation:", observation)
    
    done = False
    while not done:
        # Take a random action within the action space
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, done, info = env.step(action)
        
        # Print the current state and reward
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        env.render()  # Call render to update the graphical window
        time.sleep(0.01)  # Simulate real-time stepping
    
    env.close()

if __name__ == "__main__":
    test_hyperdog_env()
