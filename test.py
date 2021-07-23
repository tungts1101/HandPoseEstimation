import cv2
import numpy as np
import open3d
import matplotlib.pyplot as plt

# ===== read image =====
filepath = 'hand_pose_action/Video_files/Subject_1/use_calculator/1/color/color_%04d.jpeg'
# filepath = 'hand_pose_action/Video_files/Subject_1/use_calculator/1/depth/depth_%04d.png'
# # filepath = 'hand_pose_action/Video_files/Subject_1/put_salt/1/depth/depth_0006.png'

# depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

# # ===== crop all parts too far from camera =====
# depth = cv2.split(depth_img)[0]
# depth[depth>550] = 0

# depth_checked_hand_obj = depth / 256.0
# # cv2.imshow('Depth checked', depth_checked_hand_obj)
# # cv2.waitKey(0)

# # ===== find the largest contour =====
# depth_u8 = depth.astype('uint8')
# # threshold_value = int(np.max(depth_u8) * 0.1)
# _, threshold = cv2.threshold(depth_u8, 0, 255, cv2.THRESH_TRIANGLE)
# kernel = np.ones((1,1),np.uint8)
# dilate = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, 3)

# contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# c = max(contours, key = cv2.contourArea)
# x,y,w,h = cv2.boundingRect(c)

# # ===== show cropped hand and obj =====
# mask = np.zeros(depth_img.shape[0:2], dtype=np.uint8)
# cv2.drawContours(mask, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
# # cv2.imshow('Mask', mask)
# # cv2.waitKey(0)

# out = np.zeros_like(depth_img)
# out[mask == 255] = depth_img[mask == 255]

# # uncomment to view the mask of hand and object cropped by depth based and contour based
# contour_checked_hand_obj = out / 256.0
# numpy_horizontal = np.hstack((depth_checked_hand_obj, contour_checked_hand_obj))
# cv2.imshow('Cropped Hand Object', numpy_horizontal)
# cv2.waitKey(0)

## region of interest
# roi = out[y:y+h,x:x+w]

backSub = cv2.createBackgroundSubtractorMOG2()
    
capture = cv2.VideoCapture(filepath, cv2.CAP_IMAGES)

# stack = []
cur_frame = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    # fgMask = backSub.apply(frame)
    
    # # cv2.imshow('Frame', frame)
    # # cv2.imshow('FG Mask', fgMask)

    # out = np.zeros_like(frame)
    # out[fgMask > 0] = frame[fgMask > 0]
    # out[out>500] = 0

    # # stack.append(frame)

    # out = out / 256.0
    # cv2.imshow('Cropped Hand Object', out)

    # Step 2: Convert to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Step 3: Create a mask based on medium to high Saturation and Value
    # - These values can be changed (the lower ones) to fit your environment
    mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
    # We need a to copy the mask 3 times to fit the frames
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # # Step 4: Create a blurred frame using Gaussian blur
    # blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    # # Step 5: Combine the original with the blurred frame based on mask
    # frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)

    depthpath = 'hand_pose_action/Video_files/Subject_1/use_calculator/1/depth/depth_{:04d}.png'.format(cur_frame)
    depth_img = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)

    out = np.zeros_like(depth_img)
    out = np.where(mask_3d[0:2]!=(255,255), depth_img, out)
    # out[mask_3d[0:2] == (255,255)] = depth_img[mask_3d[0:2] == (255,255)]

    out = out / 256.0

    cur_frame += 1

     # Step 6: Show the frame with blurred background
    cv2.imshow("Webcam", out)

    keyboard = cv2.waitKey(0)
    if keyboard == 113:
        break

# sequence = np.stack(stack, axis=0)
# # Repace each pixel by mean of the sequence
# result = np.mean(sequence, axis=0).astype(np.uint8)

# result = result / 256.0
# cv2.imshow('stack', result)
# cv2.waitKey(0)