#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

# Workaround for the dll files
import os
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)) + "/lib")
from Error import ObException
import StreamProfile
import Pipeline
from Property import *
from ObTypes import *

from utils import CvFpsCalc
from utils import gesturecalcs
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_type", type=int, default=0)
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


def main():
    # Argument parsing #####################################################
    args = get_args()
    device_type = args.device_type
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation #####################################################
    if device_type == 0:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    else:
        pipe = Pipeline.Pipeline(None, None)
        # Configure which streams to enable or disable for the Pipeline by creating a Config
        config = Pipeline.Config()
        try:
            # Get all stream configurations of a color camera, including stream resolution, frame rate, and frame format
            profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)
            videoProfile = None
            try:
                # Find the corresponding Profile according to the specified format, the RGB888 format is preferred
                videoProfile = profiles.getVideoStreamProfile(
                    640, 0, OB_PY_FORMAT_RGB888, 30)
                print(videoProfile, "lol")
            except ObException as e:
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                    e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
                # Alternative if it does not exist
                videoProfile = profiles.getVideoStreamProfile(
                    640, 0, OB_PY_FORMAT_UNKNOWN, 30)
            colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
            config.enableStream(colorProfile)
        except ObException as e:
            print("Current device does not support color sensor!")
            sys.exit()
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get_fps()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = gesturecalcs.select_mode(key, mode)

        # Camera capture #####################################################
        if device_type == 0:
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)
        else:
            frameSet = pipe.waitForFrames(100)   
            if frameSet == None or frameSet.colorFrame() == None or frameSet.depthFrame() == None:
                continue
            else:
                colorFrame = frameSet.colorFrame()
                colorSize = colorFrame.dataSize()
                colorData = colorFrame.data()
                colorWidth = colorFrame.width()
                colorHeight = colorFrame.height()
                if colorSize != 0:
                    colorData.resize((colorHeight, colorWidth, 3))
                    colorData = cv.resize(colorData, (320, 240))
                    image = colorData # The data is already in RGB format
                    image.flags.writeable = False

                debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = gesturecalcs.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = gesturecalcs.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = gesturecalcs.pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = gesturecalcs.pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                gesturecalcs.logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = gesturecalcs.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = gesturecalcs.draw_landmarks(debug_image, landmark_list)
                debug_image = gesturecalcs.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = gesturecalcs.draw_point_history(debug_image, point_history)
        debug_image = gesturecalcs.draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
