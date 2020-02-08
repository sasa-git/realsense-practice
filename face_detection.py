import numpy as np
import cv2
import pyrealsense2 as rs

# read Classfier models
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # color to gray scale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # create rectangle
        for(x, y, w, h) in faces:
            # draw rectangle
            img = cv2.rectangle(color_image, (x, y), (x+w, y+h), (255,0,0), 2)

            # trim face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imshow('RealSense', depth_colormap)

        # if press key esc, run finished
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()