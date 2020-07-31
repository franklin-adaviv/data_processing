
import sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import rosbag
import open3d as o3d 
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import time 
import copy


def process_pose_data(file):
    # processes rosbag file for a T265 camera and creates a dcitionary object and list to output
    # Returns:
    #   pose_dict: [ timestamp --> pose ] this is a dictionary that maps time stamps to pose
    #   timestamsp = [timestamp] a list of timestamps

    t_end = time.time() + 12
    use_timer = True

    bag = rosbag.Bag(file)

    # intialize outputs
    pose_dict = dict()
    timestamps = []

    # initialize pose as a list [np.array , np.array]
    initial_xyz = np.array([0,0,0])
    initial_quat = R.from_rotvec([0,0,0]).as_quat()
    initial_pose = [initial_xyz, initial_quat]

    for message_data in bag.read_messages():
        if (time.time() > t_end) and use_timer:
            print("exceeded rosbag read time")
            break
        else: 
            #try:
            (topic, msg, t) = message_data

            # topic names
            a_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
            b_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
            c_topic = "/device_0/sensor_0/Pose_0/pose/metadata"
    

            # cases
            if topic == a_topic:

                # update timestamps
                timestamp = round(t.secs + t.nsecs*10**(-9),9)
                timestamps.append(timestamp)
                print(timestamp)

                # extract message info
                t_x = float(msg.translation.x)
                t_y = float(msg.translation.y)
                t_z = float(msg.translation.z)

                r_x = float(msg.rotation.x)
                r_y = float(msg.rotation.y)
                r_z = float(msg.rotation.z)
                r_w = float(msg.rotation.w)

                # intialize time stamps
                if len(timestamps) == 0:

                    timestamps.append(timestamp)
                    pose_dict[timestamp] = initial_pose

                # add new pose info
                new_quat = np.array([r_x,r_y,r_z,r_w])
                new_pose = np.array([t_x,t_y,t_z])
                timestamps.append(timestamp)
                pose_dict[timestamp] = [new_pose,new_quat]

                # # get previous pose info
                # prev_timestamp = timestamps[-1]
                # prev_xyz, prev_rot = pose_dict[prev_timestamp]
                
                # # get quaternion
                # r_q_prev = R.from_quat(prev_quat)
                # r_q_delta = R.from_quat([r_x,r_y,r_z,r_w])
                # r_q_curr = r_q_delta*r_q_prev
                # curr_quat = r_q_curr.asquat()

                # # get translation
                # pos_xyz_delta = np.array(t_x,t_y,t_z)



                # print(t_x,t_y,t_z,r_x,r_y,r_z,r_w)
                # print(np.linalg.norm(np.array([r_x,r_y,r_z,r_w])))

            # except:
            #   print("error in rosbag reading")

    return np.array(timestamps), pose_dict

def pointcloud_from_rgbd(file,timestamps,pose_dict):
    # processes rosbag file for a T265 camera and creates a dcitionary object and list to output
    # Returns:
    #   pose_dict: [ timestamp --> pose ] this is a dictionary that maps time stamps to pose
    #   timestamsp = [timestamp] a list of timestamps

    # intialize rosbag obj
    bag = rosbag.Bag(file)

    # set time to stop accepting new messages
    use_timer = False
    t_end = time.time() + 50

    # skips N number of image samples
    images_to_skip = 10

    # set rostopic names
    depth_topic = "/device_0/sensor_0/Depth_0/image/data"
    color_topic = "/device_0/sensor_1/Color_0/image/data"

    # stores RGB-D image pairs
    RGBD_pair = [None,None]

    # intialize point cloud object adn voxel_size
    all_cloud = o3d.geometry.PointCloud()
    voxel_size = 100

    for message_data in bag.read_messages():
        #try:
        (topic, msg, t) = message_data

        if topic == depth_topic:
            
            # convert msg into image
            cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
            cv_image = cv_image.astype(np.uint16)
            if RGBD_pair[1] is None:
                RGBD_pair[1] = cv_image

            # # visualize image ###
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', cv_image)
            # cv2.waitKey(1)

        elif topic == color_topic:

            # convert msg into image
            cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
            cv_image = cv_image.astype(np.uint16)
            
            if RGBD_pair[0] is None:
                RGBD_pair[0] = cv_image
            
            # visualize image ###
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', cv_image)
            # cv2.waitKey(1)

        ##### Process image data ####
        if RGBD_pair[0] is not None and RGBD_pair[1] is not None:
            color_image, depth_image = RGBD_pair
            image_frame_count += 1

            if image_frame_count > images_to_skip:

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
                # # width: 640, height: 480, ppx: 319.398, ppy: 237.726, fx: 381.68, fy: 381.68, model: 4, coeffs: [0, 0, 0, 0, 0]
                # # width: 640, height: 480, ppx: 335.273, ppy: 248.74, fx: 612.64, fy: 612.115, model: 2, coeffs: [0, 0, 0, 0, 0]
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                all_cloud += cloud
                all_cloud.voxel_down_sample(voxel_size = voxel_size)
        o3d.visualization.draw_geometries([all_cloud])   
        # except:
        #   pass
def visualize_pose(timestamps,pose_dict):

    # get sampled time stamp info
    N_t = np.size(timestamps)
    N_samples = 8
    sample_indices = np.arange(0,N_t,N_t//N_samples)
    sampled_timestamps = timestamps[sample_indices]
    print(N_t)
    print(sampled_timestamps)
    
    # initialize 
    intital_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    all_axes = [intital_axes]

    # create all the rotated and transalte
    for timestamp in sampled_timestamps:
        pos, quat = pose_dict[timestamp]
        mesh = copy.deepcopy(intital_axes)
        R_matrix = mesh.get_rotation_matrix_from_quaternion(quat)
        mesh.rotate(R_matrix, center = (0,0,0))
        mesh.translate(pos)
        mesh.scale(0.2,center=(0,0,0))
        all_axes.append(mesh)
    o3d.visualization.draw_geometries(all_axes)



if __name__ == "__main__":
    b1 = "data/20200728_164058.bag" # T265
    b2 = "data/20200728_164057.bag" # L515
    timestamps, pose_dict = process_pose_data(b1)
    pointcloud_from_rgbd(b2,timestamps,pose_dict)
    # visualize_pose(timestamps,pose_dict)




