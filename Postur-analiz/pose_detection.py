import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye_inner': 1,
    'right_eye_inner': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Defining global counters and states
left_arm_counter = 0
right_arm_counter = 0
previous_left_arm_status = False
previous_right_arm_status = False

# Model loading
model_handle = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(model_handle)
inference_fn = model.signatures['serving_default']

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    if angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    elif angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    angle_deg = np.degrees(angle_rad)
    return np.abs(angle_deg) if angle_deg < 180 else 360 - angle_deg

def preprocess_frame(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def run_inference(model, image):
    image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
    image = tf.cast(image, dtype=tf.int32)
    outputs = model(image)
    return outputs['output_0'].numpy()