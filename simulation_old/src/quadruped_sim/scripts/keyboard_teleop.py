import rclpy
from rclpy.node import Node
import sys
import select
import tty
import termios
from geometry_msgs.msg import Twist

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # Publisher for cmd_vel
        self.publisher = self.create_publisher(Twist, '/hyperdog/cmd_vel', 10)
        
        # Settings for non-blocking keyboard input
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
        self.get_logger().info("Keyboard Teleop Ready. Use WASD keys:")
        self.get_logger().info("W: Forward, S: Backward")
        self.get_logger().info("A: Turn Left, D: Turn Right")
        self.get_logger().info("Q: Quit")
        
        self.spin_thread()

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def spin_thread(self):
        while rclpy.ok():
            key = self.get_key()
            
            cmd = Twist()
            if key == 'w':  # Forward
                cmd.linear.x = 0.5
            elif key == 's':  # Backward
                cmd.linear.x = -0.5
            elif key == 'a':  # Turn Left
                cmd.angular.z = 0.5
            elif key == 'd':  # Turn Right
                cmd.angular.z = -0.5
            elif key == 'q':  # Quit
                rclpy.shutdown()
                break
            
            self.publisher.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    keyboard_teleop = KeyboardTeleop()
    
    try:
        rclpy.spin(keyboard_teleop)
    except KeyboardInterrupt:
        keyboard_teleop.destroy_node()
    finally:
        rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, keyboard_teleop.settings)

if __name__ == '__main__':
    main()