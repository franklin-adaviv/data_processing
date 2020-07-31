import sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import rosbag
import open3d as o3d 
from cv_bridge import CvBridge
import time 
from scipy import signal as sp
import scipy
import copy
from clustering_algorithm_lib import *
from sklearn.cluster import KMeans


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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
            o3d.registration.TransformationEstimationPointToPoint(),
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



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def ply_analyze_canopy(file):
    # parameters
    voxel_size = 15

    # get data
    cloud = o3d.io.read_point_cloud(file)
    cloud = cloud.voxel_down_sample(voxel_size = voxel_size)

    # trasform data into world frame
    z_camera_to_table = 700
    Beta = 43*2*np.pi/360

    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([-Beta,np.pi,np.pi/2]))    # for the pots file

    #R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,np.pi/2,np.pi/2]))     # for the long bench file
    cloud.rotate(R,center = np.array([0,0,0]))
    cloud.translate(np.array([0,0,z_camera_to_table]))

    # make plane points
    points = np.asarray(cloud.points)
    points[:,1] = -points[:,1]
    A = np.ones(np.shape(points)); A[:,:2] = points[:,:2]
    B = points[:,2:]
    fit, residual, rnk , s = scipy.linalg.lstsq(A,B)
    # create plane mesh
    xlim = [0,5000]
    ylim = [0,5000]
    step = 100
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1], step),
                  np.arange(ylim[0], ylim[1], step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    plane_xyz = np.stack((X,Y,Z)).transpose()
    cloud_table_plane = o3d.geometry.PointCloud()
    cloud_table_plane.points = o3d.utility.Vector3dVector(plane_xyz)
    cloud_table_plane.paint_uniform_color([1,0,0.7])

    # calculating rotation correction
    plane_normal = -np.array([fit[0,0],fit[1,0],-1])
    theta_y = np.arctan2(plane_normal[0],plane_normal[2])
    theta_x = np.arctan2(plane_normal[1],plane_normal[2])
    cloud_table_plane_points = np.asarray(cloud_table_plane.points)
    rotation_correction_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.array([theta_x,-theta_y,0]))
    cloud_table_plane.rotate(rotation_correction_matrix,center = np.array([0,0,0]))

    # calculatin translation correction vector
    z_correction = np.mean(cloud_table_plane_points[:,2])
    translation_correction_vector = np.array([0,0,-z_correction])
    cloud_table_plane.translate(translation_correction_vector)

    # rotate and translate for correction.
    cloud.rotate(rotation_correction_matrix,center = np.array([0,0,0]))
    cloud.translate(translation_correction_vector)

    # get min and max
    points = cloud.points 
    min_values = np.min(points, axis = 0)
    max_values = np.max(points, axis = 0)

    # choose x to start at 0
    points = ((points - np.array([min_values[0],min_values[1],0]))/voxel_size).astype(int)
    points_map_dim = np.ceil((max_values - min_values)/voxel_size).astype(int)
    points_map_grid = np.zeros(points_map_dim)

    # create density array. 2d array with values being the avg z of the x,y position
    density_arr = []
    for x in range(points_map_dim[0]):
        row_arr = []
        for y in range(points_map_dim[1]):
            row_arr.append([])
        density_arr.append(row_arr)
    for i in range(len(points[:,0])):
        x = points[i,0]
        y = points[i,1]
        z = points[i,2]
        density_arr[x][y].append(z)
    for x in range(points_map_dim[0]):
        for y in range(points_map_dim[1]):
            if len(density_arr[x][y]) == 0:
                density_arr[x][y] = 0
            else:
                density_arr[x][y] = np.mean(density_arr[x][y])
    density_arr = np.array(density_arr)
    
    # Create a 1d array with values being avg z. We do this by first getting creating lists of all z values, then computing the avg
    arr = []
    for ix in range(np.max(points[:,0])+1):
        arr.append([])
    for ix in range(len(points[:,0])):
        x_coord = points[ix,0]
        z_coord = points[ix,2]
        arr[x_coord].append(z_coord)
    for ix in range(len(arr)):
        if len(arr[ix]) == 0:
            arr[ix] = 0
            print("this shouldn't happen")
        else: 
            arr[ix] = np.mean(arr[ix])
    arr = np.array(arr)

    plt.figure()
    plt.subplot(121)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.imshow(np.flip(density_arr.transpose()))
    plt.colorbar()
    plt.subplot(222)
    plt.stem(arr,use_line_collection = True)
    plt.ylabel("the avg z value of points")
    plt.xlabel("x value")
    plt.subplot(224)
    plt.xlabel("k values")
    plt.ylabel("FFT coeff")
    FFT = np.fft.fft(arr)
    plt.stem(np.abs(FFT),use_line_collection = True)
    plt.show()

    # # axes 
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1000, origin = np.array([0,0,0]))

    # create visualization
    o3d.visualization.draw_geometries([cloud, cloud_table_plane, axes], point_show_normal = False) 
def rotate_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def remove_distortion(cloud):
    # the SLAM algorithm introduces a dsitortion. while the points should fit a line in the xy plane,
    # the errors in pose estimation cause the points to form a curve in the xy plane.
    # this function takes in a point cloud and tries to project the best fit spline of the points in xy plane onto a line.
    # returns a new pointcloud object.
    
    # extract point cloud info
    points = np.asarray(cloud.points) 
    colors = np.asarray(cloud.colors)
    x = points[:,0]
    y = points[:,1]

    # find best fit line
    m,b = np.polyfit(points[:,0],points[:,1], 1)

    # align points with x axis
    y = y - b # set b = 0.
    theta = np.arctan2(m,1)
    new_x, new_y = rotate_2d((0,0),(x,y),-theta)
    new_points = np.stack((new_x,new_y)).transpose()
    print(np.shape(new_points))

    # find best fit curve
    print("finding curve")
    sampled = new_points[np.random.randint(new_points.shape[0],size = 500), :]
    sampled = sampled[np.argsort(sampled[:,0])]
    sampled_x = sampled[:,0]
    sampled_y = sampled[:,1]
    f_curve = np.poly1d(np.polyfit(sampled_x, sampled_y, 3))
    curve_x = sampled_x
    curve_y = f_curve(curve_x)
    print("done")

    # change_y 
    proj_x = new_x
    proj_y = new_y - f_curve(new_x)
    proj_z = points[:,2]
    proj_points = np.stack((proj_x,proj_y,proj_z)).transpose()


    plt.figure()
    plt.subplot(221)
    plt.scatter(x,y)
    plt.plot(x, m*x+b)
    plt.subplot(222)
    plt.scatter(sampled_x,sampled_y)
    plt.plot(curve_x,curve_y)
    plt.subplot(223)
    ix = np.random.randint(proj_points.shape[0],size = 100)
    sampled_proj_x = proj_points[ix,0]
    sampled_proj_y = proj_points[ix,1]
    plt.scatter(sampled_proj_x,sampled_proj_y)
    plt.savefig("output/remove_distort.png")

    proj_cloud = o3d.geometry.PointCloud()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1000, origin = np.array([0,0,0]))
    proj_cloud.points = o3d.utility.Vector3dVector(proj_points)
    proj_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([proj_cloud,axes])

    return proj_cloud

def show_ply_file(file):
    # parameters
    voxel_size = 15

    # get data
    cloud = o3d.io.read_point_cloud(file)
    cloud = cloud.voxel_down_sample(voxel_size = voxel_size)
    print(cloud.points)
    # trasform data into world frame
    z_camera_to_table = 850
    Beta = 35*2*np.pi/360
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi/2 - Beta,np.pi,np.pi/2]))
    cloud.rotate(R,center = np.array([0,0,0]))
    cloud.translate(np.array([0,0,z_camera_to_table]))

    # extract data
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    # extract z near 0 points, corresponding to points near table
    height_threshold = 200
    mask_table = np.where((points[:,2] < height_threshold) & (points[:,2] > -height_threshold ))
    cloud_table = o3d.geometry.PointCloud()
    cloud_table.points = o3d.utility.Vector3dVector(points[mask_table])
    cloud_table.colors = o3d.utility.Vector3dVector(colors[mask_table])
    #cloud_table.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=20))
    
    # calculating the table plane
    points = np.asarray(cloud_table.points)
    A = np.ones(np.shape(points)); A[:,:2] = points[:,:2]
    B = points[:,2:]
    fit, residual, rnk , s = scipy.linalg.lstsq(A,B)

    # creating a table points mesh point cloud
    xlim = [0,5000]
    ylim = [0,5000]
    step = 100
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1], step),
                  np.arange(ylim[0], ylim[1], step))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    plane_xyz = np.stack((X,Y,Z)).transpose()
    cloud_table_plane = o3d.geometry.PointCloud()
    cloud_table_plane.points = o3d.utility.Vector3dVector(plane_xyz)
    cloud_table_plane.paint_uniform_color([1,0,0.7])

    # calculating rotation correction
    plane_normal = -np.array([fit[0,0],fit[1,0],-1])
    theta_y = np.arctan2(plane_normal[0],plane_normal[2])
    theta_x = np.arctan2(plane_normal[1],plane_normal[2])
    cloud_table_plane_points = np.asarray(cloud_table_plane.points)
    rotation_correction_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.array([theta_x,-theta_y,0]))
    cloud_table_plane.rotate(rotation_correction_matrix,center = np.array([0,0,0]))

    # calculatin translation correction vector
    z_correction = np.mean(cloud_table_plane_points[:,2])
    translation_correction_vector = np.array([0,0,-z_correction])
    cloud_table_plane.translate(translation_correction_vector)

    # rotate and translate for correction.
    cloud.rotate(rotation_correction_matrix,center = np.array([0,0,0]))
    cloud.translate(translation_correction_vector)


    # Do curved distortion removal 
    cloud = remove_distortion(cloud)
    
    # create mask for pots
    pot_height = 375; pot_threshold = 200
    pot_points = np.asarray(cloud.points)
    pot_colors = np.asarray(cloud.colors)
    pot_mask = np.where( (pot_points[:,2] > pot_height - pot_threshold)&(pot_points[:,2] < pot_height + pot_threshold) )
    pot_points = pot_points[pot_mask]
    pot_colors = pot_colors[pot_mask]
    cloud_pot = o3d.geometry.PointCloud()
    cloud_pot.points = o3d.utility.Vector3dVector(pot_points)
    cloud_pot.colors = o3d.utility.Vector3dVector(pot_colors)

    # create occupany grid for pots
    pots_xy = pot_points[:,:2]
    pots_xy_min = np.min(pots_xy, axis = 0)
    pots_xy_max = np.max(pots_xy, axis = 0)
    pots_xy_dim = np.ceil((pots_xy_max - pots_xy_min)/voxel_size).astype(int)
    pots_map_grid = np.zeros(pots_xy_dim)
    pots_map_points = ((pots_xy - pots_xy_min)/voxel_size).astype(int)
    pots_map_grid[pots_map_points[:,0], pots_map_points[:,1]] = 1

    # use clustering to segment the occupancy grid
    cl = Cluster_Labeling(pots_map_grid)  

    pot_edge_len = 250/voxel_size 
    pot_area = pot_edge_len**2
    large_clusters = [tup for tup in cl.sorted_clusters if (tup[1] > pot_area*.75)]

    # process large clusters and segment them so that its N pots rather than 1 large pot
    new_labels = np.zeros(np.shape(cl.labels))
    for cluster_id, cluster_size in large_clusters:
        if cluster_size > pot_area*1.5:
            # kmeans to segment
            num_pots = int(cluster_size/pot_area)
            print(cluster_size, cluster_size/pot_area, num_pots)
            cluster_points = np.array([[tup[0],tup[1]] for tup in cl.assignments[cluster_id]])
            model = KMeans(n_clusters = num_pots, random_state = 0 ).fit(cluster_points)
            labels = model.labels_; labels += 1
            cluster_centers = model.cluster_centers_
            inertia = model.inertia_

            # create new cluster ids for each pot
            for new_ix in range(num_pots):
                new_labels[cluster_points[:,0],cluster_points[:,1]] = cluster_id + .5**labels 


        else:
            cluster_points = np.array([[tup[0],tup[1]] for tup in cl.assignments[cluster_id]])
            new_labels[cluster_points[:,0],cluster_points[:,1]] = cluster_id


    plt.close("all")
    plt.figure()
    plt.subplot(211)
    plt.imshow(np.flip(cl.labels.transpose(),axis = 0))
    plt.subplot(212)
    plt.imshow(np.flip(new_labels.transpose(),axis = 0))
    plt.savefig("output/pots.png")
    plt.close("all")

    # outlier removal
    cloud_pot = copy.deepcopy(cloud_pot)
    cl, ind = cloud_pot.remove_statistical_outlier(nb_neighbors=20,std_ratio=1.0)
    # display_inlier_outlier(cloud_pot, ind)
    cloud_pot = cloud_pot.select_by_index(ind)

    # # axes 
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1000, origin = np.array([0,0,0]))

    # create visualization
    o3d.visualization.draw_geometries([cloud, cloud_table_plane, axes], point_show_normal = False)


if __name__ == "__main__":

    ### PLY files ###
    f1 = "data/4_pots_optimized.ply"
    f2 = "data/sample_GR(optimized).ply"
    f3 = "data/pot_full.ply"
    show_ply_file(f3)
    #ply_analyze_canopy(f2)
    ### Bag Files ###

    b1 = "data/20200724_134822.bag"
    b2 = "data/20200724_135552.bag"
    # show_rosbag(b2)

