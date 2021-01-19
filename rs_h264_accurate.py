import pyrealsense2 as rs
import numpy as np
import cv2

# Config params
rgb_frame_width = 1280
rgb_frame_height = 720
depth_frame_width = 848
depth_frame_height = 480
capture_FPS = 30
write_FPS = 10
color_path = 'rs_rgb.mp4'
depth_path = 'rs_depth.mp4'


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, depth_frame_width, depth_frame_height, rs.format.z16, capture_FPS)
config.enable_stream(rs.stream.color, rgb_frame_width, rgb_frame_height, rs.format.bgr8, capture_FPS)

colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'avc1'), write_FPS, (rgb_frame_width, rgb_frame_height), 1)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'avc1'), write_FPS, (depth_frame_width, depth_frame_height), 0)

pipeline.start(config)

depth_unit = 0

frameCtr = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth values to float
        if frameCtr == 0:
            depth_unit = depth_frame.get_units()
        depth_image = depth_image * depth_unit
        #print(depth_image[0][500])

        # Round-off with precision=1 and multiply by 10, to ensure 2-digit number (CV_8U compatible)
        depth_image = (np.around(depth_image, 1) * 10).astype(np.uint8)
        #depth_image = (depth_image * 10).astype(np.uint8)
        #print(depth_image[0][500])
        #print(depth_image.dtype)
        #print(depth_image.shape)
        #break

        color_image = np.asanyarray(color_frame.get_data())
        #print(color_image.dtype)
        #print(color_image.shape)
        #break
        #print(depth_image[col][row])

        #print(depth_image.dtype)
        #break

        colorwriter.write(color_image)
        depthwriter.write(depth_image)

        cv2.imshow('Stream', depth_image)
        
        if cv2.waitKey(1) == ord("q"):
            break

        frameCtr += 1

finally:
    colorwriter.release()
    depthwriter.release()
    pipeline.stop()
