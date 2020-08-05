
import sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import rosbag
import open3d as o3d 
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import time 
from data_reader import *
import copy

def show_rosbag(file):
    bag = rosbag.Bag(file)
    bridge = CvBridge()

    # camera state is a vector [ xyz_pos, xyz_vel, xyz_accel, theta_xyz_pos, theta_xyz_vel]
    dts = np.array([])
    timestamps = np.array([])
    camera_states = np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],dtype = float)
    prev_timestamp = 0
    dt = 0

    # RBGD image pair
    RGBD_pair = [None,None]

    is_accel_updated = False
    is_gyro_updated = False

    # show 3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # point cloud scene
    image_frame_count = 0
    images_to_skip = 100
    all_pcd = []

    for message_data in bag.read_messages():
        try:
            (topic,msg,t) = message_data


            # topic names
            depth_topic = "/device_0/sensor_0/Depth_0/image/data" 
            gyro_topic = "/device_0/sensor_2/Gyro_0/imu/data"
            color_topic = "/device_0/sensor_1/Color_0/image/data"
            accel_topic = "/device_0/sensor_2/Accel_0/imu/data"

            # get prev camera state
            prev_camera_state = camera_states[-1]

            if topic == depth_topic:
                
                # convert msg into image
                cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                cv_image = cv_image.astype(np.uint16)
                print(cv_image[300:310, 300:302])
                if RGBD_pair[1] is None:
                    RGBD_pair[1] = cv_image

                # visualize image ###
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', cv_image)
                cv2.waitKey(1)

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

            elif topic == accel_topic:

                # Header
                frame_id = msg.header.frame_id
                secs, nsecs = msg.header.stamp.secs, msg.header.stamp.nsecs
                timestamp = secs + nsecs/1000000000.0
                # Accel vector
                accel  = np.array([msg.linear_acceleration.x , msg.linear_acceleration.y , msg.linear_acceleration.z]) - np.array([-0.13030262513798302, -9.49842453250673, 0.7248717691083202])
                # update
                prev_camera_state[2] = accel
                is_accel_updated = True

            elif topic == gyro_topic:
                # Header
                frame_id = msg.header.frame_id
                secs, nsecs = msg.header.stamp.secs, msg.header.stamp.nsecs
                timestamp = secs + nsecs/1000000000.0 
                # ang Vel Vector
                ang_vel = np.array([msg.angular_velocity.x , msg.angular_velocity.y , msg.angular_velocity.z])
                # update
                prev_camera_state[4] = ang_vel
                is_gyro_updated = True

            else:
                pass
            ### RGBD image ###
            if RGBD_pair[0] is not None and RGBD_pair[1] is not None:

                color_image, depth_image = RGBD_pair
                image_frame_count += 1

                if image_frame_count > images_to_skip:

                    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
                    # # width: 640, height: 480, ppx: 319.398, ppy: 237.726, fx: 381.68, fy: 381.68, model: 4, coeffs: [0, 0, 0, 0, 0]
                    # # width: 640, height: 480, ppx: 335.273, ppy: 248.74, fx: 612.64, fy: 612.115, model: 2, coeffs: [0, 0, 0, 0, 0]
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                    all_pcd.append(pcd)

            ### update position ###
            if is_gyro_updated and is_accel_updated:
                if prev_timestamp == 0:
                    prev_timestamp = timestamp
                
                # set flags to false
                is_gyro_updated = False
                is_accel_updated = False

                # calc dt
                dt = timestamp - prev_timestamp
                prev_timestamp = timestamp
                timestamps = np.append(timestamps,timestamp)
                dts = np.append(dts,dt)

                # get camera vectors
                camera_state = camera_states[-1]
                camera_pos = camera_state[0]
                camera_vel = camera_state[1]
                camera_accel = camera_state[2]
                camera_ang_pos = camera_state[3]
                camera_ang_vel = camera_state[4]

                # update
                new_camera_pos = camera_pos + camera_vel*dt
                new_camera_vel = camera_vel + camera_accel*dt 
                new_camera_ang_pos = camera_ang_pos + camera_ang_vel*dt 

                new_camera_state = np.array([[new_camera_pos,new_camera_vel,camera_accel,new_camera_ang_pos,camera_ang_vel]],dtype = float)
                camera_states = np.concatenate((camera_states,new_camera_state))
        except:
            pass

    ########## After the For Loop ############

    #### run 3d visualization ###
    for ix in range(1,len(all_pcd)):
        pcd = all_pcd[ix]
        if ix != 1:
            source_pcd = all_pcd[ix]
            target_pcd = all_pcd[ix-1]
            voxel_size = 10
            result_global = execute_global_registration(source_pcd,target_pcd,voxel_size)

            trans_matrix_global_reg = result_global.transformation
            threshold = voxel_size*1.5
            result_local = execute_local_registration(source_pcd,target_pcd,trans_matrix_global_reg,threshold)

            source_pcd.transform(result_local.transformation)

    o3d.visualizer.draw_geometries(all_pcd)


    # x = camera_states[:,0,0]
    # y = camera_states[:,0,1]
    # z = camera_states[:,0,2]

    # v_x = camera_states[:,1,0]
    # v_y = camera_states[:,1,1]
    # v_z = camera_states[:,1,2]

    # a_x = camera_states[:,2,0]
    # a_y = camera_states[:,2,1]
    # a_z = camera_states[:,2,2]

    # calibration = [np.mean(v) for v in (a_x,a_y,a_z)]

    # ### Post process the camera state data ###

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(x,y)
    # plt.title("xy position")
    # plt.subplot(222)
    # plt.plot(v_x,'g')
    # plt.plot(a_x,'r')
    # plt.title("x")
    # plt.subplot(224)
    # plt.plot(v_y,'g')
    # plt.plot(a_y,'r')
    # plt.title("y")
    # plt.show()

def process_pose_data(file):
    # processes rosbag file for a T265 camera and creates a dcitionary object and list to output
    # Returns:
    #   pose_dict: [ timestamp --> pose ] this is a dictionary that maps time stamps to pose
    #   timestamsp = [timestamp] a list of timestamps

    t_end = 11
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


        #try:
        (topic, msg, t) = message_data
        timestamp = round(t.secs + t.nsecs*10**(-9),9)

        # end early
        print(timestamp,t_end)
        if use_timer and (timestamp > t_end):
            print("exceeded rosbag read time")
            break


        # topic names
        a_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
        b_topic = "/device_0/sensor_0/Pose_0/pose/transform/data"
        c_topic = "/device_0/sensor_0/Pose_0/pose/metadata"


        # cases
        if topic == a_topic:

            # update timestamps
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

    # intialize rosbag file and bridge
    bag = rosbag.Bag(file)
    bridge = CvBridge()

    # set time to stop accepting new messages
    use_timer = True
    t_end = 10

    # skips N number of image samples
    image_frame_count = 0
    image_step_size = 30

    # set image info
    image_size = [480,640]
    depth_topic = "/device_0/sensor_0/Depth_0/image/data"
    color_topic = "/device_0/sensor_1/Color_0/image/data"

    # stores RGB-D image pairs
    RGBD_pair = [None,None]

    # offsett
    trans_offset = None
    # intialize point cloud object adn voxel_size
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axes.scale(1,center = (0,0,0))
    all_cloud = o3d.geometry.PointCloud()
    voxel_size = 1
    print("starting to read RGBD bag: ")
    for message_data in bag.read_messages():

        #try:
        (topic, msg, t) = message_data

        # calc timestamp
        timestamp = round(t.secs + t.nsecs*10**(-9),9)
        if use_timer and (timestamp > t_end):
            break

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
            cv_image = cv2.resize(cv_image.astype(np.uint8),(image_size[1],image_size[0]))

            
            if RGBD_pair[0] is None:
                RGBD_pair[0] = cv_image          
            # #visualize image ###
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', cv_image)
            # cv2.waitKey(1)

        ##### Process image data ####
        if RGBD_pair[0] is not None and RGBD_pair[1] is not None:
            print(timestamp," point cloud frame received")
            color_image, depth_image = RGBD_pair
            image_frame_count += 1
            RGBD_pair = [None,None]



            if image_frame_count%image_step_size == 0:

                # extract point cloud from images
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
                # # width: 640, height: 480, ppx: 319.398, ppy: 237.726, fx: 381.68, fy: 381.68, model: 4, coeffs: [0, 0, 0, 0, 0]
                # # width: 640, height: 480, ppx: 335.273, ppy: 248.74, fx: 612.64, fy: 612.115, model: 2, coeffs: [0, 0, 0, 0, 0]
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
                
                # transform point cloud
                pose_time = timestamps[np.argmin(timestamps - timestamp)]
                pos_arr, quat = pose_dict[pose_time]
                R_matrix = axes.get_rotation_matrix_from_quaternion(quat)
                cloud.rotate(R_matrix, center = (0,0,0))
                if trans_offset is not None:
                    #cloud.transform(trans_offset)
                    cloud.translate(pos_arr)

                    # add point cloud to all point clouds
                    all_cloud += cloud
                    #all_cloud.voxel_down_sample(voxel_size = voxel_size)
                
                prev_cloud = cloud

            if trans_offset is None and image_frame_count > 1 + 2*image_step_size:
                threshold = voxel_size*1.5
                result_global = execute_global_registration(cloud, prev_cloud, voxel_size)
                trans_init = result_global.transformation
                result_local = execute_local_registration(cloud, prev_cloud,trans_init,threshold)
                trans_offset = result_local.transformation
                draw_registration_result(cloud, prev_cloud,trans_offset)


    all_cloud.voxel_down_sample(voxel_size = voxel_size)
    
    o3d.visualization.draw_geometries([all_cloud,axes])   

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
    # b1 = "data/20200728_164058.bag" # T265
    # b2 = "data/20200728_164057.bag" # L515
    # timestamps, pose_dict = process_pose_data(b1)
    # pointcloud_from_rgbd(b2,timestamps,pose_dict)
    # # visualize_pose(timestamps,pose_dict)


    ### Bag Files ###
    b1 = "data/20200724_134822.bag"
    b2 = "data/20200724_135552.bag"
    # show_rosbag(b2)



