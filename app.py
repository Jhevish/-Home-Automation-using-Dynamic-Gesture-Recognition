#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time
import serial
from gpiozero import PWMLED

from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Argument parser to handle user inputs for camera settings and detection parameters
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

# Toggle device power state (on/off) and provide haptic feedback
def device_power_switch(device_status, feedback_device):
    if device_status:
        print("APP_off")
        feedback_device.value = 1  # Short buzz
        time.sleep(0.5)
        feedback_device.value = 1  # Medium buzz
        time.sleep(1)
        feedback_device.value = 0  # Tiny stop
        time.sleep(0.55)
        feedback_device.value = 1  # Long buzz
        time.sleep(2)
        feedback_device.value = 0  # Stop buzzing 
    else:
        print("Activated")
        feedback_device.value = 1  # Long buzz
        time.sleep(2)
        feedback_device.value = 0  # Tiny stop
        time.sleep(0.55)
        feedback_device.value = 1  # Medium buzz
        time.sleep(1)
        feedback_device.value = 0  # Tiny stop
        time.sleep(0.55)
        feedback_device.value = 1  # Short buzz
        time.sleep(1)
        feedback_device.value = 0  # Stop buzzing 
    return not device_status

# Function to handle device control logic based on gestures
def let_arduino(input, current_levels, device_number, hf_device, feedback_):
    if not feedback_:
        return current_levels
    max_appliance_level = 4
    min_appliance_level = 0

    device_prefix = 'L' if device_number == 0 else 'F'

    # Increase appliance intensity
    if input == 0:
        if current_levels[device_number] == max_appliance_level:
            return current_levels
        print("letscome")
        current_levels[device_number] = min(4, current_levels[device_number] + 1)
    # Decrease appliance intensity
    elif input == 1:
        if current_levels[device_number] == min_appliance_level:
            return current_levels
        print("letsgo")
        current_levels[device_number] = max(0, current_levels[device_number] - 1)
    # Turn off the appliance
    elif input == 2:
        if current_levels[device_number] == min_appliance_level:
            return current_levels
        print("Over")
        current_levels[device_number] = 0

    # Send the updated intensity level to the Arduino
    send_signal_to_arduino(f"{device_prefix}{current_levels[device_number]}\n", serial_device_led)
    hf_device.value = 1  # Short buzz
    time.sleep(1.0)
    hf_device.value = 0  # Stop buzz
    return current_levels

# Sends control signals to the Arduino via serial communication
def send_signal_to_arduino(signal, device_serial):
    device_serial.write(signal.encode())
    print(f"Actuated: {signal}")

# Handles mode selection based on key input
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # Numbers 0 to 9
        number = key - 48
    if key == 110:  # 'n' key for mode 0
        mode = 0
    if key == 107:  # 'k' key for mode 1
        mode = 1
    if key == 104:  # 'h' key for mode 2
        mode = 2
    return number, mode

# Calculate the bounding rectangle around the detected hand
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# Convert hand landmarks into a list of coordinates
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Preprocess hand landmarks for gesture classification
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten the list of landmarks
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize the landmark values
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Preprocess point history for dynamic gesture classification
def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Flatten the list of point history
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

# Logs the gesture data (landmarks and point history) to a CSV file
def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

# Draw detected hand landmarks on the frame
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Draw circles on key points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Base of thumb
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb (proximal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb (distal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger (proximal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger (middle)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger (distal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger (proximal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger (middle)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger (distal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger (proximal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger (middle)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger (distal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Pinky (proximal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Pinky (middle)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Pinky (distal)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Pinky tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

# Draw bounding rectangle around detected hand
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Draw rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

# Display hand gesture and finger gesture information on the frame
def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image

# Draw the history of finger tip points on the frame
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

# Display FPS and mode information on the frame
def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

# Main function to handle camera capture, gesture detection, and device control
def main():
    # Initialize serial communication with Arduino
    serial_device_led = serial.Serial('/dev/ttyACM0', 9600)

    # Parse arguments for camera and detection settings
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True  # Use bounding rectangle around detected hand

    # Setup GPIO for haptic feedback device (e.g., vibration motor)
    hf_device = PWMLED(14)

    # Gesture counters for tracking various gestures
    boost_up = 0
    unboost_device = 0
    power_toggle = 0
    device_off = 0
    light_switch = 0
    fan_switch = 0

    # Initialize device control variables
    device_level = False
    intensity_power = [0, 0]  # Device power levels (light and fan)
    led_character = 0  # Represents the LED/light device
    character_fan = 1  # Represents the fan device
    device_active_pointer = led_character  # Tracks which device is currently active

    # Setup and start the camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={'size': (cap_width, cap_height), 'format': 'RGB888'},  # or BGR888
        controls={'FrameRate': 50},  # will default to 30fps otherwise
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    # Load Mediapipe's hand detection model with specified settings
    hands_detec = mp.solutions.hands
    hands = hands_detec.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Load custom gesture classification models
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Load gesture labels from CSV files
    with open('/home/jhevish/Downloads/THEOHANDGESTURE30JULY - Copy/hand-gesture-recognition-mediapipe-main/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    with open('/home/jhevish/Downloads/THEOHANDGESTURE30JULY - Copy/hand-gesture-recognition-mediapipe-main/model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    # Initialize FPS calculator for measuring the frame rate
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Initialize deques to store hand landmarks and gesture history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Main loop to capture frames, detect gestures, and control devices
    mode = 0

    while True:
        fps = cvFpsCalc.get()  # Calculate the current frame rate

        key = cv.waitKey(10)  # Check for key press to terminate the program (ESC key)
        if key == 27:  # ESC key to break the loop
            break
        number, mode = select_mode(key, mode)

        # Capture a frame from the camera
        request = picam2.capture_request()
        image = request.make_array("main")
        request.release()

        # Flip the camera image for correct display
        image = cv.flip(image, 0)

        # Create a copy of the image for debugging purposes
        debug_image = copy.deepcopy(image)

        # Convert the image to RGB format for Mediapipe processing
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)  # Perform hand detection
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate bounding rectangle around the hand
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Extract hand landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks and point history for classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                # Log the landmarks and history to a CSV file if in logging mode
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # Classify the hand sign (static gesture)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture detected
                    point_history.append(landmark_list[8])  # Store the index finger tip position
                else:
                    point_history.append([0, 0])

                # Classify dynamic gestures (e.g., motion gestures)
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Track gesture history and get the most common gesture
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Control devices based on the recognized dynamic gesture
                moving_gesture = point_history_classifier_labels[most_common_fg_id[0][0]]

                if moving_gesture == "Clockwise":
                    boost_up += 1
                    unboost_device = 0
                    if boost_up == 20:
                        intensity_power = let_arduino(0, intensity_power, device_active_pointer, hf_device, True)
                        boost_up = 0
                elif moving_gesture == "Counter Clockwise":
                    unboost_device += 1
                    boost_up = 0
                    if unboost_device == 20:
                        intensity_power = let_arduino(1, intensity_power, device_active_pointer, hf_device, True)
                        unboost_device = 0
                else:
                    boost_up = 0
                    unboost_device = 0

                # Control device power states based on static gestures
                static_gesture = keypoint_classifier_labels[hand_sign_id]

                # Toggle device power using "Open" gesture
                if static_gesture == "Open":
                    power_toggle += 1
                    if power_toggle == 30:
                        dmc_status = device_power_switch(True, hf_device)
                        power_toggle = -10
                else:
                    power_toggle = 0

                # Turn off device using "Close" gesture
                if static_gesture == "Close":
                    device_off += 1
                    if device_off == 5:
                        intensity_power = let_arduino(2, intensity_power, device_active_pointer, hf_device, True)
                        device_off = 0
                else:
                    device_off = 0

                # Switch to control the light using "TwoFingers" gesture
                if static_gesture == "TwoFingers":
                    light_switch += 1
                    fan_switch = 0
                    if light_switch == 15:
                        device_active_pointer = led_character
                        light_switch = 0
                else:
                    light_switch = 0

                # Switch to control the fan using "ThreeFingers" gesture
                if static_gesture == "ThreeFingers":
                    fan_switch += 1
                    light_switch = 0
                    if fan_switch == 15:
                        device_active_pointer = character_fan
                        fan_switch = 0
                else:
                    fan_switch = 0

                # Draw the bounding rectangle, landmarks, and gesture information on the frame
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], point_history_classifier_labels[most_common_fg_id[0][0]])
        else:
            # Reset gesture counters if no hand is detected
            point_history.append([0, 0])
            power_toggle = 0
            boost_up = 0
            unboost_device = 0
            device_off = 0
            light_switch = 0
            fan_switch = 0

        # Draw point history and other information on the frame
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Display the frame
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Cleanup: release resources and ensure devices are turned off
    serial_device_led.write(b'L0\n')  # Turn off the LED/light
    serial_device_led.write(b'F0\n')  # Turn off the fan
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
