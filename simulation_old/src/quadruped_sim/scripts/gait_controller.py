import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import math

class QuadrupedGaitController(Node):
    def __init__(self):
        super().__init__('quadruped_gait_controller')
        
        # Gait parameters
        self.gait_types = {
            'walk': self.walk_gait,
            'trot': self.trot_gait,
            'crawl': self.crawl_gait
        }
        
        # Subscribers and publishers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/hyperdog/cmd_vel', self.gait_callback, 10
        )
        
        # Joint publishers (you'll need to adapt to your specific robot)
        self.joint_publishers = {}
        
    def gait_callback(self, msg):
        # Determine gait based on velocity
        speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        
        if speed < 0.2:
            gait = 'crawl'
        elif speed < 0.5:
            gait = 'walk'
        else:
            gait = 'trot'
        
        # Execute selected gait
        self.gait_types[gait](msg)
    
    def walk_gait(self, cmd_vel):
        """Quadruped walking gait implementation"""
        # Basic walking pattern
        # This is a simplified model - real implementation requires precise leg coordination
        phase_offset = [0, math.pi/2, math.pi, 3*math.pi/2]
        
        # Calculate joint angles for each leg
        for leg in range(4):
            swing_height = 0.05  # 5cm lift
            stride_length = cmd_vel.linear.x * 0.2
            
            # Simple sinusoidal leg movement
            t = rclpy.time.Time.now().nanoseconds / 1e9
            angle = math.sin(t + phase_offset[leg]) * swing_height
            
            # Publish joint angles (placeholder - adapt to your URDF)
            # self.joint_publishers[f'leg_{leg}'].publish(angle)
    
    def trot_gait(self, cmd_vel):
        """Diagonal gait where opposite diagonal legs move together"""
        # Similar to walk, but with different phase offsets
        # More rapid, efficient movement
        pass
    
    def crawl_gait(self, cmd_vel):
        """Slow, precise movement for obstacle navigation"""
        # Each leg moves independently, very slow progression
        pass

def main(args=None):
    rclpy.init(args=args)
    gait_controller = QuadrupedGaitController()
    rclpy.spin(gait_controller)
    gait_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()