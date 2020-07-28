import sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import rosbag
import open3d as o3d 
from cv_bridge import CvBridge
import time 
from scipy import signal as sp


def execute_global_registration(source_pcd, target_pcd, voxel_size,distance_threshold = None):
    
    def preprocess_point_cloud(pcd, voxel_size):
        # The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point.
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 3
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=20))
        return pcd_down, pcd_fpfh
    distance_threshold = voxel_size*1.5 if distance_threshold == None else distance_threshold
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result
def execute_local_registration(source_pcd,target_pcd,trans_init,threshold):
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))

    reg_p2p = o3d.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPlane(),
            o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    return reg_p2p

def get_array(fname):
    arr = np.loadtxt(open(fname, "rb"), delimiter=" ")/1000
    return arr



def show_arr():
    fname = "depth_image.csv"
    im = get_array(fname)
    print(np.shape(im))
    plt.imshow(im,vmax = 200)
    plt.show()

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
                cv_image = cv_image.astype(np.uint8)
                if RGBD_pair[1] is None:
                    RGBD_pair[1] = cv_image

                ## visualize image ###
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('RealSense', cv_image)
                # cv2.waitKey(1)

            elif topic == color_topic:

                # convert msg into image
                cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                cv_image = cv_image.astype(np.uint8)
                
                if RGBD_pair[0] is None:
                    RGBD_pair[0] = cv_image
                
                # ## visualize image ###
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

            
def show_ply_file(file):

    # get data
    cloud = o3d.io.read_point_cloud(file)
    cloud = cloud.voxel_down_sample(voxel_size = 25)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=20))

    # trasform data into world frame
    z_camera_to_table = 850
    Beta = 43*2*np.pi/360
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi/2 - Beta,np.pi,np.pi/2]))
    cloud.rotate(R,center = np.array([0,0,0]))
    cloud.translate(np.array([0,0,z_camera_to_table]))

    # extract data
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    colors = np.asarray(cloud.colors)

    # extract z near 0 points, corresponding to points near table
    height_threshold = 200
    mask_table = np.where((points[:,2] < height_threshold) & (points[:,2] > -height_threshold ))
    cloud_table = o3d.geometry.PointCloud()
    cloud_table.points = o3d.utility.Vector3dVector(points[mask_table])
    cloud_table.colors = o3d.utility.Vector3dVector(colors[mask_table])
    cloud_table.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=20))
    
    # axes 
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1000, origin = np.array([0,0,0]))

    # create visualization
    o3d.visualization.draw_geometries([cloud_table, axes], point_show_normal = True) 



if __name__ == "__main__":

    ### PLY files ###
    f1 = "data/4_pots_optimized.ply"
    show_ply_file(f1)
    ### Bag Files ###
    b1 = "data/20200724_134822.bag"
    #show_rosbag(b1)  

