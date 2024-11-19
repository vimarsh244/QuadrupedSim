import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
p.connect(p.GUI)  # Use p.DIRECT for headless mode
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the Plane and Mini Pupper URDF
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("/home/vimarsh/Desktop/ROS-Stuff/QuadrupedSim/QuadrupedSim/PyBullet_RL/pupper_env/urdf/pupper.urdf", basePosition=[0, 0, 0.2])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)

# Visualize Mini Pupper
while True:
    p.stepSimulation()
    time.sleep(0.01)


# Get joint info
num_joints = p.getNumJoints(robot_id)
print(f"Number of Joints: {num_joints}")

# Set initial joint positions
for joint in range(num_joints):
    p.setJointMotorControl2(
        robot_id,
        joint,
        p.POSITION_CONTROL,
        targetPosition=0,
        force=5
    )

# Test joint movements
for _ in range(500):
    for joint in range(num_joints):
        p.setJointMotorControl2(
            robot_id,
            joint,
            p.POSITION_CONTROL,
            targetPosition=0.5 * (-1 if joint % 2 == 0 else 1),
            force=10
        )
    p.stepSimulation()
    time.sleep(0.01)

