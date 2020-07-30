
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
			a_topic = "something"
			b_topic = "something"
			
			if topic == a_topic:
				# get pose 


		except:
			pass



if __name__ == "__main__":

