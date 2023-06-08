# Workaround for the dll files
import os
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)) + "/../lib")

from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import imutils
import numpy as np
import sys
from pythonosc import osc_message_builder
from pythonosc import udp_client
from collections import deque
import math

np.set_printoptions(threshold=sys.maxsize)
q = 113
ESC = 27

# ======= OSC UDP Parameters =======
ipAddress = '127.0.0.1'
port = 8888
header = "/orbbec"
# Create an OSC client
client = udp_client.SimpleUDPClient(ipAddress, port)


params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False

params.filterByArea = True
params.minArea = 50  # define min area
params.maxArea = 500  # define max area
ver = (cv2.__version__).split('.')

# constants to monitor touches
OBJECT_DEPTH = 22
# this is unavoidable, the camera is not super high quality
OBJECT_MARGIN = 5
FINGER_DEPTH = 15
FINGER_MARGIN = 5
RADIUS_LOWER = 10
RADIUS_UPPER = 20
MIN_DIST = 10000
FREE_LOCATION = (-1000, -1000)
pts = []
buffer = 64

def euclidean_diff(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def find_closest_pairs(existing_contours, coords2):
    num_existing = len(existing_contours)
    num_new = len(coords2)
    
    if num_existing == 0 or num_new == 0:
        return ([], [i for i in range(num_existing)], [j for j in range(num_new)])
    
    dist_matrix = np.zeros((num_existing, num_new)) + np.inf
    
    for i in range(num_existing):
        for j in range(num_new):
            dist_matrix[i,j] = euclidean_diff(existing_contours[i][1], coords2[j])
    
    pairs = []
    while np.min(dist_matrix) != np.inf:
        idx = np.argmin(dist_matrix)
        i, j = np.unravel_index(idx, dist_matrix.shape)
        if i < num_existing and j < num_new:
            pairs.append((existing_contours[i][0], j))
        dist_matrix[i,:] = np.inf
        dist_matrix[:,j] = np.inf
    
    unmatched_existing = [i for i in range(num_existing) if i not in [pair[0] for pair in pairs]]
    unmatched_new = [j for j in range(num_new) if j not in [pair[1] for pair in pairs]]
    return pairs, unmatched_existing, unmatched_new


def find_closest_pairs_LK(existing_contours, coords2, prev_frame, current_frame):
    num_existing = len(existing_contours)
    num_new = len(coords2)
    
    if num_existing == 0 or num_new == 0:
        return ([], [i for i in range(num_existing)], [j for j in range(num_new)])
    
    prev_gray = prev_frame
    curr_gray = current_frame
    
    # Calculate optical flow using Lucas-Kanade method
    prev_pts = np.array([contour[1] for contour in existing_contours], dtype=np.float32)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    pairs = []
    for i in range(num_existing):
        if status[i] == 1:
            closest_idx = np.argmin(np.linalg.norm(next_pts[i] - coords2, axis=1))
            pairs.append((existing_contours[i][0], closest_idx))
    
    unmatched_existing = [i for i in range(num_existing) if i not in [pair[0] for pair in pairs]]
    unmatched_new = [j for j in range(num_new) if j not in [pair[1] for pair in pairs]]
    return pairs, unmatched_existing, unmatched_new


def find_closest_pairs_F(existing_contours, coords2, prev_frame, current_frame):
    num_existing = len(existing_contours)
    num_new = len(coords2)
    
    if num_existing == 0 or num_new == 0:
        return ([], [i for i in range(num_existing)], [j for j in range(num_new)])
    
    prev_gray = prev_frame
    curr_gray = current_frame
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    pairs = []
    for i in range(num_existing):
        x, y = existing_contours[i][1]
        dx, dy = flow[int(y), int(x)]
        new_point = (x + dx, y + dy)
        distances = np.linalg.norm(np.array(coords2) - new_point, axis=1)
        closest_idx = np.argmin(distances)
        pairs.append((existing_contours[i][0], closest_idx))
    
    unmatched_existing = [i for i in range(num_existing) if i not in [pair[0] for pair in pairs]]
    unmatched_new = [j for j in range(num_new) if j not in [pair[1] for pair in pairs]]
    return pairs, unmatched_existing, unmatched_new

if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

try:
    # Create a Pipeline, which is the entry point of the API, which can be easily opened and closed through the Pipeline
    # Multiple types of streams and get a set of frame data
    pipe = Pipeline.Pipeline(None, None)
    # Configure which streams to enable or disable for the Pipeline by creating a Config
    config = Pipeline.Config()

    windowsWidth = 0
    windowsHeight = 0
    try:
        # Get all stream configurations of the depth camera, including stream resolution, frame rate, and frame format
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)
        videoProfile = None
        try:
            # Find the corresponding Profile according to the specified format(Y16 format is preferred)
            videoProfile = profiles.getVideoStreamProfile(
                640, 0, OB_PY_FORMAT_Y12, 30)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            # If Y16 format is not found, the format does not match and the corresponding Profile is searched for open stream
            videoProfile = profiles.getVideoStreamProfile(
                640, 0, OB_PY_FORMAT_UNKNOWN, 30)
        depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        windowsWidth = depthProfile.width()
        windowsHeight = depthProfile.height()
        config.enableStream(depthProfile)
    except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
            e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("Current device is not support depth sensor!")
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
    depthFrame = frameSet.depthFrame()
    if depthFrame != None:
        size = depthFrame.dataSize()
        data = depthFrame.data()

        if size != 0:
            # Resize frame data to (height,width,2)
            data.resize((windowsHeight, windowsWidth, 2))
            # mirror image
            data = np.flip(data, 1)
            # Convert frame for processing
            newData = data[:, :, 0]+data[:, :, 1]*256
            # Convert frame for 8 bit display
            newDataRender = newData.astype(np.uint8)
            maxDepth = np.max(newData)
            # save this as the initialised background depth
            bg = newData

    while True:
        # Wait for a frame of data in a blocking manner. This frame is a composite frame that contains frame data of all streams enabled in the configuration.
        # And set the frame waiting timeout to 100ms
        frameSet = pipe.waitForFrames(100)
        if frameSet == None:
            continue
        else:
            # render depth frame
             # Depth Objects Recognition ###########################################################
            depthFrame = frameSet.depthFrame()
            if depthFrame != None:
                size = depthFrame.dataSize()
                data = depthFrame.data()

                if size != 0:
                    # reformat data
                    data.resize((windowsHeight, windowsWidth, 2))
                    # data = np.flip(data, 1)
                    ## exclusive to this setup
                    data = np.flip(data, 0)
                    newData = data[:, :, 0]+data[:, :, 1]*256
                    filteredData = bg - newData
                    filteredData[filteredData > maxDepth] = 0
                    filteredDataRender = filteredData.astype(np.uint8)

                    # obtain the filtered circles(on the first level)
                    blurredCircles = cv2.GaussianBlur(filteredData, (7, 7), 0)
                    thresholdCircles = cv2.inRange(
                        blurredCircles, OBJECT_DEPTH-OBJECT_MARGIN, OBJECT_DEPTH+OBJECT_MARGIN)
                    # perform dilations and erosions to remove small blobs left in the mask
                    maskCircles = cv2.erode(
                        thresholdCircles, None, iterations=2)
                    
                    # find contours in the mask and initialize the current
                    # (x, y) centre of the ball
                    contoursObject = cv2.findContours(maskCircles.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                    contoursObject = imutils.grab_contours(contoursObject)
                    contours = []
  
                    maskDataRGB = cv2.cvtColor(maskCircles, cv2.COLOR_GRAY2RGB)
                    # remove contours that are the wrong size
                    for i in range(len(contoursObject)-1, -1, -1):
                        ((x, y), radius) = cv2.minEnclosingCircle(
                            contoursObject[i])
                        if radius < RADIUS_LOWER or radius > RADIUS_UPPER:
                            del contoursObject[i]
                        else:
                            M = cv2.moments(contoursObject[i])
                            try:
                                centre = (int(M["m10"] / M["m00"]),
                                      int(M["m01"] / M["m00"]))
                            except ZeroDivisionError:
                                centre = (0, 0)
                            cv2.circle(maskDataRGB, centre, 5, (0, 0, 255), -1)
                            contours.append([int(radius), centre])
                    existing_contours = [(item["id"], item["pts"][-1]) for i,item in enumerate(pts) if len(item["pts"])>0]
                    centres = [sublist[1] for sublist in contours if sublist]
                    (pairs, unmatched_existing, unmatched_new) = find_closest_pairs(
                        existing_contours, centres)
                    
                    # Match existing pairs
                    for pair in pairs:
                        id = pair[0]
                        centre = centres[pair[1]]
                        label = "Circle {}".format(id)
                        pts_dict = next(item for item in pts if item['id'] == id)
                        pts_list = pts_dict['pts']
                        cv2.circle(maskDataRGB, centre, 20,
                                            (255, 0, 0), 2)
                        

                        if len(pts_list):
                            cv2.putText(maskDataRGB, label, centre,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                    # remove objects that no longer exist
                    for item in pts:
                        id = item['id']
                        pts_list = item['pts']
                        if id not in [pair[0] for pair in pairs]:
                            pts_list.clear()

                    # add new objects to the deque
                    for unmatched in unmatched_new:
                        centre = centres[unmatched]
                        deque_available = False
                        for existing_pts in pts:
                            if len(existing_pts['pts']) == 0:
                                id = existing_pts['id']
                                pts_list = existing_pts['pts']
                                pts_list.appendleft(centre)
                                deque_available = True
                                break
                        # if all of the points list are in use, add new one
                        if not deque_available:
                            id = len(pts)
                            new_pts = {'id': id, 'pts': deque(maxlen=buffer)}
                            new_pts['pts'].appendleft(centre)
                            pts.append(new_pts)
                        label = "Circle {}".format(id)
                        if (id >= 0):
                            pts_dict = next(item for item in pts if item['id'] == id)

                        cv2.putText(maskDataRGB, label, centre,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow("tracker", maskDataRGB)
                    key = cv2.waitKey(1)
                    # Press ESC or 'q' to close the window
                    if key == ESC or key == q:
                        cv2.destroyAllWindows()
                        break
    pipe.stop()

except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
        e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
