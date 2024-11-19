import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity

def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('quadruped_sim')
    
    # Path to URDF
    urdf_path = os.path.join(pkg_share, 'urdf', 'hyperdog.urdf')
    
    # Gazebo launch
    gazebo_launch = ExecuteProcess(
        cmd=['gazebo', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Spawn robot node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'hyperdog', 
            '-file', urdf_path,
            '-x', '0', 
            '-y', '0', 
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_robot
    ])