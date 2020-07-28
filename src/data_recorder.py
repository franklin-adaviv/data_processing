# test lidar
# test max time 
# test smaller April tags
# compare april tags and without april tags
#
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d



def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

# start streaming pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Configure Stream for IMU data
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)


# set record file to filename.bag
config.enable_record_to_file('image_with_imu_other.bag')

# Start streaming
pipeline_profile = pipeline.start(config)

# record device
device_record = pipeline_profile.get_device()
device_recorder = device_record.as_recorder()

# initilize timing objects. Will alternate and record for 5 sec. and pause for 3
t_event_start = cv2.getTickCount()
t_event_switch = cv2.getTickCount()
total_time_secs = 30
record_time_secs = 65
pause_time_secs = 0

# Initialize State
state = "record"

try:

    while True:
        ### Compute Time ###
        t_now = cv2.getTickCount()
        t_total = (t_now - t_event_start) / cv2.getTickFrequency()
        t_switch = (t_now - t_event_switch) / cv2.getTickFrequency()

        ### check end criteria ###
        if t_total > total_time_secs: # change it to record what length of video you are interested in
            print("Done!")
            break
        ### manage states ###
        if state == "record":
            if t_switch > record_time_secs:
                print("starting pause")
                state = "pause"
                t_event_switch = cv2.getTickCount()
                rs.recorder.pause(device_recorder)
        elif state == "pause":
            if t_switch > pause_time_secs:
                print("starting record")
                state = "record"
                t_event_switch = cv2.getTickCount()
                rs.recorder.resume(device_recorder)

        ### do action ###
        if state == "record":
            print("recording ", t_total)
            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            motion_frame = frames.as_motion_frame()
            pose_frame = frames.get_pose_frame()

            if not depth_frame or not color_frame:
                continue

            # extract the accel and gyro as index 2 and 3. The order of config matters
            accel = accel_data(frames[2].as_motion_frame().get_motion_data())
            gyro = gyro_data(frames[3].as_motion_frame().get_motion_data())
            #print("accel: ", accel ,"  gyro: ", gyro)
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            print(type(depth_image),np.shape(depth_image),depth_image.dtype)

            # Apply colormap on depth image (image must be converted to8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)

            ##### Visualizing 
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # ### estimate pose 
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            # rs.extrinsics.translation
            # rs.extrinsics.rotation



            # ## convert depth images to colored 3d point ###
            # pc = rs.pointcloud()
            # points = pc.calculate(depth_frame)
            # pc.map_to(color_frame)
            # pc_data = points.data 
            # vtx = np.asanyarray(points.get_vertices())
            # tex = np.asanyarray(points.get_texture_coordinates())

            # npy_vtx = np.zeros((len(vtx), 3), float)
            # for i in range(len(vtx)):
            #     npy_vtx[i][0] = np.float(vtx[i][0])
            #     npy_vtx[i][1] = np.float(vtx[i][1])
            #     npy_vtx[i][2] = np.float(vtx[i][2])
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(npy_vtx)
            # o3d.visualization.draw_geometries([pcd])

            # npy_tex = np.zeros((len(tex), 3), float)
            # for i in range(len(tex)):
            #     npy_tex[i][0] = np.float(tex[i][0])
            #     npy_tex[i][1] = np.float(tex[i][1])
            ############## Below owkrsb
            # p = rs.pose()
            # print(p)
            # ##########
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
            # o3d.camera.PinholeCameraIntrinsic(
            #     o3d.camera.PinholeCameraIntrinsicParameters.
            #     PrimeSenseDefault))

            # o3d.visualization.draw_geometries([pcd])

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
            #######
        elif state == "pause":
            print("paused ", t_total)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)
            cv2.namedWindow('RealSense_paused', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense_paused', depth_colormap)
            cv2.waitKey(1)
            

finally:
    pass
    # Stop streaming
    pipeline.stop()