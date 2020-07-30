
import sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import rosbag
import open3d as o3d 
from cv_bridge import CvBridge
import time 

def process_pose_data(file):
	# processes rosbag file for a T265 camera and creates a dcitionary object and list to output
	# Returns:
	#	pose_dict: [ timestamp --> pose ] this is a dictionary that maps time stamps to pose
	#	timestamsp = [timestamp] a list of timestamps

	bag = rosbag.Bag(file)


	pose = np.array([0,0,0,0,0,0])
	for message_data in bag.read_messages():
		try:
			(topic, msg, t) = message_data

			# topic names
			a_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
			b_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
			c_topic = "/device_0/sensor_0/Pose_0/pose/metadata"
				
			# time
			#print(t)
			timestamp = t

			# cases
			if topic == a_topic:
				# get pose 
				#print(msg)
				t_x = float(msg.translation.x)
				t_y = float(msg.translation.y)
				t_z = float(msg.translation.z)

				r_x = float(msg.rotation.x)
				r_y = float(msg.rotation.y)
				r_z = float(msg.rotation.z)
				r_w = float(msg.rotation.w)

				print(t_x,t_y,t_z,r_x,r_y,r_z,r_w)
				print(np.linalg.norm(np.array([r_x,r_y,r_z,r_w])))

			elif topic == b_topic:
				#print(msg)
				pass
			elif topic == c_topic:
				#print(msg)
				pass


		except:
			pass



if __name__ == "__main__":
	b1 = "data/20200728_164058.bag"
	process_pose_data(b1)

