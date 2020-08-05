#!/usr/bin/env python
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import Accel
from sensor_msgs.msg import Imu

class IMU_republisher():
    def __init__(self):
        
        self.lin_accel = None
        self.ang_vel = None

        # init node
        rospy.init_node('imu_listener')

        # create pushlisher for IMU data
        self.pub = rospy.Publisher('/device_0/sensor_2/imu/imu/data',Imu,queue_size = 1)
        
        # initialize subscribers
        lin_accel_sub = rospy.Subscriber('/device_0/sensor_2/Accel_0/imu/data',Imu, self.update_lin_accel_callback) 
        rot_vel_sub = rospy.Subscriber('/device_0/sensor_2/Gyro_0/imu/data',Imu, self.update_rot_vel_callback)
    
        # finish initialization
        # rospy.Rate(0.5)

    #def publish_callback(self):
    #    if self.ang_vel is not None and self.lin_accel is not None:
    #        imu = self.imu
    #        imu.linear_acceleration = self.lin_accel
    #        self.pub.publish(imu)
    
    def update_lin_accel_callback(self,data):
        self.lin_accel = data.linear_acceleration
        if self.rot_vel is not None:
            imu = data
            #print("lin_accel <<<<<<<<<<<<<")
            #print(self.lin_accel)
            #print("rot_vel <<<<<<<<<<<")
            #print(self.rot_vel)
            #print(imu.angular_velocity,"angvel")
            imu.angular_velocity = self.rot_vel
            self.pub.publish(imu)

    def update_rot_vel_callback(self,data):
        self.rot_vel = data.angular_velocity
        if self.lin_accel is not None:
            
            imu = data
            #print("lin accel")
            #print(self.lin_accel)
            #print("rot_vel")
            #print(self.rot_vel)
            #print(imu.linear_acceleration,"linaccel")
            imu.linear_acceleration = self.lin_accel
            print(type(self.rot_vel),type(data.angular_velocity),"AV")
            print(type(self.lin_accel),type(imu.linear_acceleration))
            self.pub.publish(imu)
        
if __name__ == "__main__":
    IMU_republisher()
    rospy.spin()
