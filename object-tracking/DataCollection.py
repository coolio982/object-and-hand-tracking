# Workaround for the dll files
import os
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)) + "/../lib")

import time
from pythonosc import udp_client
from pythonosc import osc_message_builder
import configparser
import csv
import sys
import numpy as np
import imutils
import cv2
from Error import ObException
import StreamProfile
import Pipeline
from Property import *
from ObTypes import *



np.set_printoptions(threshold=sys.maxsize)
q = 113
ESC = 27


# stores centre of objects as {(x, y)}
centres = []
buffer = 64  # buffer of where the object has been
# Callback function for mouse events


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Left button down: x={}, y={}'.format(x, y))
        pixel_value = data[y, x]
        distance = pixel_value[0]+pixel_value[1]*256
        difference = filteredData[y, x]
        print(f'Pixel value at ({x}, {y}): {pixel_value}')
        print(f'Distance  at ({x}, {y}): {distance} mm')
        print(f'Difference at ({x}, {y}): {difference} mm')
        centres.append((x, y-40))
    elif event == cv2.EVENT_RBUTTONDOWN:
        print('Saving Image...')
        cv2.imwrite(f'Images/DepthImage/frame_{int(time.time())}.jpg', filteredData)


def calibrate_background(data, height, width):
    # Resize frame data to (height,width,2)
    data.resize((height, width, 2))
    # mirror image
    data = np.flip(data, 1)
    # Convert frame for processing
    newData = data[:, :, 0]+data[:, :, 1]*256
    maxDepth = np.max(newData)
    # save this as the initialised background depth
    return newData, maxDepth

def exit_program():
    cv2.destroyAllWindows()
    # save the ground truth centres
    with open('./Images/groundTruth.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(centres)


try:
    # Create a Pipeline, which is the entry point of the API, which can be easily opened and closed through the Pipeline
    # Multiple types of streams and get a set of frame data
    pipe = Pipeline.Pipeline(None, None)
    # Configure which streams to enable or disable for the Pipeline by creating a Config
    config = Pipeline.Config()

    windowsWidth = 0
    windowsHeight = 0
    try:
        # Get all stream configurations of a color camera, including stream resolution, frame rate, and frame format
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)
        videoProfile = None
        try:
            # Find the corresponding Profile according to the specified format, the RGB888 format is preferred
            videoProfile = profiles.getVideoStreamProfile(
                640, 0, OB_PY_FORMAT_RGB888, 30)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            # Alternative if it does not exist
            videoProfile = profiles.getVideoStreamProfile(
                640, 0, OB_PY_FORMAT_UNKNOWN, 30)
        colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        config.enableStream(colorProfile)
    except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("Current device does not support color sensor!")

    try:
        # Get depth camera stream information
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

        videoProfile = None
        try:
            # Y16 format is preferred
            videoProfile = profiles.getVideoStreamProfile(640,0,OB_PY_FORMAT_Y16,30)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            # Alternative if Y16 not found
            videoProfile = profiles.getVideoStreamProfile(640,0,OB_PY_FORMAT_UNKNOWN,30)
        depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        config.enableStream(depthProfile)
    except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("Current device does not support depth sensor!")
        sys.exit()

    # Start the flow configured in Config, if no parameters are passed, the default configuration startup flow will be started
    pipe.start(config, None)

    # Get whether the mirror attribute has writable permission
    if pipe.getDevice().isPropertySupported(OB_PY_PROP_DEPTH_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
        # set mirror
        pipe.getDevice().setBoolProperty(OB_PY_PROP_DEPTH_MIRROR_BOOL, True)
    # calibrate in a blocking manner
    # Wait for a frame of data in a blocking manner. This frame is a composite frame that contains frame data of all streams enabled in the configuration.
    # And set the frame waiting timeout to 100ms
    frameSet = pipe.waitForFrames(100)
    # render depth frame
    depthFrame = None
    while depthFrame == None:
        frameSet = pipe.waitForFrames(100)
        depthFrame = frameSet.depthFrame()
    if depthFrame != None:
        size = depthFrame.dataSize()
        data = depthFrame.data()
        depthWidth = depthFrame.width()
        depthHeight = depthFrame.height() 
        if size != 0:
            bg, maxDepth = calibrate_background(
                data, depthHeight, depthWidth)
            cv2.imwrite(f'Images/DepthImage/frame_{int(time.time())}.jpg', bg)
    print("INITIALISED")
    sys.stdout.flush()  
    while True:
        # Wait for a frame of data in a blocking manner. This frame is a composite frame that contains frame data of all streams enabled in the configuration.
        # And set the frame waiting timeout to 100ms
        frameSet = pipe.waitForFrames(100)
        if frameSet == None:
            continue
        else:
            # render depth frame
            depthFrame = frameSet.depthFrame()
            if depthFrame != None:
                size = depthFrame.dataSize()
                data = depthFrame.data()
                depthWidth = depthFrame.width()
                depthHeight = depthFrame.height() 
                if size != 0:
                    # Resize frame data to (height,width,2)
                    data.resize((depthHeight, depthWidth, 2))
                    # mirror image
                    data = np.flip(data, 1)
                    # Convert frame for processing
                    newData = data[:, :, 0]+data[:, :, 1]*256
                    # Convert frame for 8 bit display
                    newDataRender = newData.astype(np.uint8)
                    # Convert frame data GRAY to RGB
                    newDataRGB = cv2.cvtColor(
                        newDataRender, cv2.COLOR_GRAY2RGB)

                    filteredData = bg-newData
                    filteredData[filteredData > maxDepth] = 0
                    # Convert frame for 8 bit display
                    filteredDataRender = filteredData.astype(np.uint8)
                    # Convert frame data GRAY to RGB
                    filteredDataRGB = cv2.cvtColor(
                        filteredDataRender, cv2.COLOR_GRAY2RGB)
                    # create window
                    cv2.namedWindow("Difference Depth Viewer",
                                    cv2.WINDOW_NORMAL)
                    # display image
                    cv2.imshow("Difference Depth Viewer", filteredDataRGB)
                    cv2.setMouseCallback(
                        "Difference Depth Viewer", mouse_callback)
                
                    key = cv2.waitKey(1)
                    # Press ESC or 'q' to close the window
                    if key == ESC or key == q:
                        exit_program()
                        break
            # render colour frame
            colorFrame = frameSet.colorFrame()
            if colorFrame != None:
                size = colorFrame.dataSize()
                data = colorFrame.data()

                if size != 0:
                    # Resize frame data to (height,width,2)
                    colorWidth = colorFrame.width()
                    colorHeight = colorFrame.height()
                    data.resize((colorHeight, colorWidth, 3))
                    # mirror image
                    # data = np.flip(data, 1)
                    dataRender = data.astype(np.uint8)
                    # create window
                    cv2.namedWindow("Colour Viewer",
                                    cv2.WINDOW_NORMAL)
                    # display image
                    cv2.imshow("Colour Viewer", dataRender)
                    cv2.setMouseCallback(
                        "Colour Viewer", mouse_callback)
                
                    key = cv2.waitKey(1)
                    # Press ESC or 'q' to close the window
                    if key == ESC or key == q:
                        exit_program()
                        break
    pipe.stop()

except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
        e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
