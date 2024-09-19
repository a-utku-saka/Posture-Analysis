import cv2
import numpy as np
from pose_detection import calculate_angle

# Global counters and states
left_arm_counter = 0
right_arm_counter = 0
previous_left_arm_status = False
previous_right_arm_status = False

def draw_keypoints_and_edges_press(frame, keypoints, confidence_threshold=0.4):
    global left_arm_counter, right_arm_counter, previous_left_arm_status, previous_right_arm_status

    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    keypoints_dict = {}
    for idx, (ky, kx, kp_conf) in enumerate(shaped):
        if kp_conf > confidence_threshold:
            keypoints_dict[idx] = (int(kx), int(ky))

    if all(k in keypoints_dict for k in [5, 7, 9]):
        angle_left = calculate_angle(keypoints_dict[5], keypoints_dict[7], keypoints_dict[9])
        is_left_arm_up = keypoints_dict[5][1] > keypoints_dict[7][1]
        color_left = (0, 255, 0) if 135 <= angle_left <= 180 and is_left_arm_up else (0, 0, 255)
        cv2.line(frame, keypoints_dict[5], keypoints_dict[7], color_left, 2)
        cv2.line(frame, keypoints_dict[7], keypoints_dict[9], color_left, 2)
        if 135 <= angle_left <= 180 and is_left_arm_up and not previous_left_arm_status:
            left_arm_counter += 1
            previous_left_arm_status = True
        elif not (135 <= angle_left <= 180 and is_left_arm_up):
            previous_left_arm_status = False

    if all(k in keypoints_dict for k in [6, 8, 10]):
        angle_right = calculate_angle(keypoints_dict[6], keypoints_dict[8], keypoints_dict[10])
        is_right_arm_up = keypoints_dict[6][1] > keypoints_dict[8][1]
        color_right = (0, 255, 0) if 135 <= angle_right <= 180 and is_right_arm_up else (0, 0, 255)
        cv2.line(frame, keypoints_dict[6], keypoints_dict[8], color_right, 2)
        cv2.line(frame, keypoints_dict[8], keypoints_dict[10], color_right, 2)
        if 135 <= angle_right <= 180 and is_right_arm_up and not previous_right_arm_status:
            right_arm_counter += 1
            previous_right_arm_status = True
        elif not (135 <= angle_right <= 180 and is_right_arm_up):
            previous_right_arm_status = False

    cv2.putText(frame, f"Left Arm Count: {left_arm_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Arm Count: {right_arm_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_keypoints_and_edges_press_raise(frame, keypoints, confidence_threshold=0.4):
    global left_arm_counter, right_arm_counter, previous_left_arm_status, previous_right_arm_status

    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    keypoints_dict = {}
    for idx, (ky, kx, kp_conf) in enumerate(shaped):
        if kp_conf > confidence_threshold:
            keypoints_dict[idx] = (int(kx), int(ky))

    if all(k in keypoints_dict for k in [5, 7, 11]):
        angle_left = calculate_angle(keypoints_dict[5], keypoints_dict[7], keypoints_dict[11])
        is_left_arm_valid = 40 <= angle_left <= 80
        color_left = (0, 255, 0) if is_left_arm_valid else (0, 0, 255)
        cv2.line(frame, keypoints_dict[5], keypoints_dict[7], color_left, 2)
        cv2.line(frame, keypoints_dict[7], keypoints_dict[11], color_left, 2)
        if is_left_arm_valid and not previous_left_arm_status:
            left_arm_counter += 1
            previous_left_arm_status = True
        elif not is_left_arm_valid:
            previous_left_arm_status = False

    if all(k in keypoints_dict for k in [6, 8, 12]):
        angle_right = calculate_angle(keypoints_dict[6], keypoints_dict[8], keypoints_dict[12])
        is_right_arm_valid = 40 <= angle_right <= 80
        color_right = (0, 255, 0) if is_right_arm_valid else (0, 0, 255)
        cv2.line(frame, keypoints_dict[6], keypoints_dict[8], color_right, 2)
        cv2.line(frame, keypoints_dict[8], keypoints_dict[12], color_right, 2)
        if is_right_arm_valid and not previous_right_arm_status:
            right_arm_counter += 1
            previous_right_arm_status = True
        elif not is_right_arm_valid:
            previous_right_arm_status = False

    cv2.putText(frame, f"Left Arm Count: {left_arm_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Right Arm Count: {right_arm_counter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def draw_keypoints_for_hammer_curl(frame, keypoints, confidence_threshold=0.4):
    global left_arm_counter, right_arm_counter, previous_left_arm_status, previous_right_arm_status

    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    keypoints_dict = {}
    for idx, (ky, kx, kp_conf) in enumerate(shaped):
        if kp_conf > confidence_threshold:
            keypoints_dict[idx] = (int(kx), int(ky))

    if all(k in keypoints_dict for k in [5, 7, 9]):
        angle_left = calculate_angle(keypoints_dict[5], keypoints_dict[7], keypoints_dict[9])
        if 0 <= angle_left <= 40:
            if not previous_left_arm_status:
                left_arm_counter += 1
                previous_left_arm_status = True
            color_left = (0, 255, 0)
        else:
            previous_left_arm_status = False
            color_left = (0, 0, 255)
        cv2.line(frame, keypoints_dict[5], keypoints_dict[7], color_left, 2)
        cv2.line(frame, keypoints_dict[7], keypoints_dict[9], color_left, 2)

    if all(k in keypoints_dict for k in [6, 8, 10]):
        angle_right = calculate_angle(keypoints_dict[6], keypoints_dict[8], keypoints_dict[10])
        if 0 <= angle_right <= 40:
            if not previous_right_arm_status:
                right_arm_counter += 1
                previous_right_arm_status = True
            color_right = (0, 255, 0)
        else:
            previous_right_arm_status = False
            color_right = (0, 0, 255)
        cv2.line(frame, keypoints_dict[6], keypoints_dict[8], color_right, 2)
        cv2.line(frame, keypoints_dict[8], keypoints_dict[10], color_right, 2)

    cv2.putText(frame, f"Left Arm Count: {left_arm_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Arm Count: {right_arm_counter}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame