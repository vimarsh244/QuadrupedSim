
# setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'quadruped_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # Include URDF
        (os.path.join('share', package_name, 'urdf'), 
         glob('urdf/*.urdf')),
        # Include config files
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vimarsh',
    maintainer_email='vimarsh244@gmail.com',
    description='Hyperdog Gazebo Control',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gait_controller = quadruped_sim.scripts.gait_controller:main',
            'keyboard_teleop = quadruped_sim.scripts.keyboard_teleop:main',
        ],
    },
)