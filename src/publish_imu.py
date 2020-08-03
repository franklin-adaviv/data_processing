#!/usr/bin/env python
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import Accel
from sensor_msg.msg import Imu
from geometry_msgs.msg import linear_acceleration

class IMU_republisher():
    def __init__(self):
        
        self.lin_accel = A
        self.rot_vel = B

        # init node
        rospy.init_node('imu_listener')

        # create pushlisher for IMU data
        self.pub = rospy.Publisher('/device_0/sensor_2/imu/imu/data',Imu,queue_size = 1)
        
        # initialize subscribers
        lin_accel_sub = rospy.Subscriber('/device_0/sensor_2/Accel_0/imu/data') 
        rot_vel_sub = rospy.Subscriber('/device_0/sensor_2/Gyro_0/imu/data')
    
        # finish initialization
        # rospy.Rate(0.5)

    def publish_callback(self):
        imu = Imu()
        imu.linear_acceleration = self.lin_accel
        imu.angular_velocity = self.rot_vel
        self.pub.publish(imu)
    
    def update_lin_accel_callback(self,msg):
        self.lin_accel = msg.data.linear_acceleration
        self.publish_callback()
    
    def update_rot_vel_callback(self,msg):
        self.rot_vel = msg.data.angular_velocity
        self.publish_callback()

if __name__ == "__main__":
    IMU_republisher()
    rospy.spin()
