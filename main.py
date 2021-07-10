import cv2
import numpy as np
import open3d

# ===== read image =====
depth_img = cv2.imread('hand_pose_action/Video_files/Subject_1/put_salt/1/depth/depth_0001.png', cv2.IMREAD_UNCHANGED)

# ===== crop all parts too far from camera =====
depth = cv2.split(depth_img)[0]
depth[depth>550] = 0

depth_checked_hand_obj = depth / 256.0
# cv2.imshow('Depth checked', depth_checked_hand_obj)
# cv2.waitKey(0)

# ===== find the largest contour =====
depth_u8 = depth.astype('uint8')
threshold_value = int(np.max(depth_u8) * 0.1)
_, threshold = cv2.threshold(depth_u8, threshold_value, 255, cv2.THRESH_BINARY)
kernel = np.ones((1,1),np.uint8)
dilate = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, 3)

contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)

# ===== show cropped hand and obj =====
mask = np.zeros(depth_img.shape[0:2], dtype=np.uint8)
cv2.drawContours(mask, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
# cv2.imshow('Mask', mask)
# cv2.waitKey(0)

out = np.zeros_like(depth_img)
out[mask == 255] = depth_img[mask == 255]

contour_checked_hand_obj = out / 256.0
# cv2.imshow('Cropped Hand and Obj', contour_checked_hand_obj)
# cv2.waitKey(0)

numpy_horizontal = np.hstack((depth_checked_hand_obj, contour_checked_hand_obj))
# cv2.imshow('Cropped Hand Object', numpy_horizontal)
# cv2.waitKey(0)

# roi = out[y:y+h,x:x+w]

color_img = open3d.io.read_image('hand_pose_action/Video_files/Subject_1/put_salt/1/color/color_0001.jpeg')
point_cloud = open3d.geometry.RGBDImage.create_from_color_and_depth(color_img, out)