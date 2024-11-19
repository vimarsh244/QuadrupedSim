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
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("path/to/mini_pupper.urdf", basePosition=[0, 0, 0.2])

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
