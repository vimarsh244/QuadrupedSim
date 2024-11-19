from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    keyboard_teleop = Node(
        package='quadruped_sim',
        executable='keyboard_teleop',
        name='keyboard_teleop',
        output='screen'
    )

    return LaunchDescription([
        keyboard_teleop
    ])