#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

""" Lucas Kanade Optical Flow
    and Dense Optical Flow

    You can select both of them from
    command line.
"""

# Idea from: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# http://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/Lucas-Kanade2.pdf
# https://www.youtube.com/watch?v=1r8E9uAcn4E

# TODO: https://nanonets.com/blog/optical-flow/
# Check: Horn–Schunck method and Buxton–Buxton method

import cv2
import numpy as np

from utilities import VideoController
from cv2_tools.Selection import SelectorCV2


# -----------------------------

import sys

confidence = 0.5
threshold= 0.3
labelsPath = '/media/fernando/WindowsHardDisk/CommonDocuments/GitHub/tfm-artificial-vision/darknet/cfg/coco.names'

LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
weightsPath = '/media/fernando/WindowsHardDisk/CommonDocuments/GitHub/tfm-artificial-vision/darknet/yolov3.weights'
configPath = '/media/fernando/WindowsHardDisk/CommonDocuments/GitHub/tfm-artificial-vision/darknet/cfg/yolov3.cfg'
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
np.set_printoptions(threshold=sys.maxsize)

# -----------------------------


# From now on, OF means Optical Flow

class LucasKanade_OF(VideoController):
    '''
    Conceptual idea:
        - We give some points to track
        - We recive the optical flow vectors of those points.
        This can solve small motions, but: How to solve larget motions?
            - Let's use pyramids: When we go up in the pyramid, small motions
                are removed and large motions become small motions.
                So by applying Lucas-Kanade there, we get optical flow along with the scale.
    _____________________________________________

    We have seen an assumption before, that all the neighbouring pixels will have similar motion.

    Lucas-Kanade method takes a 3x3 patch around the point.
    So all the 9 points have the same motion.

    We get (fx, fy, ft) for these 9 points.
    Now we need to solve 9 equations with two unknown variables.

    A better solution is obtained with least square fit method. 
    '''

    def __init__(self, video, stream, fps, scale=1, separate_frame=None, concatenate_frame=None, **kwargs):
        VideoController.__init__(self, video, stream, fps)

        max_corners = 200

        # Parameters for ShiTomasi corner detection
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        # Create some random colors
        self.color = np.random.randint(0, 255, (max_corners, 3))

        self.scale = scale
        self.separate_frame = separate_frame
        self.concatenate_frame = concatenate_frame


    def __set_win_size(self, x):
        if x < 5 or x > 50:
            return
        self.lk_params['winSize'] = (x + 5, x + 5)

    
    def __set_max_level(self, x):
        if x < 1 or x > 10:
            return
        self.lk_params['maxLevel'] = x + 1

    
    def run(self):
        gray_frame = None
        current_points = None
        mask = None
        window_name = 'Lucas Kanade'

        cv2.namedWindow(window_name)
        cv2.createTrackbar('WinSize+5', window_name, 0, 45, self.__set_win_size)
        cv2.createTrackbar('Maxlevel+1', window_name, 0, 20, self.__set_max_level)

        for frame in self.manager_cv2:
            prev_gray_frame = gray_frame

            if self.separate_frame is not None:
                frame, not_processable_frame = self.separate_frame(frame, fx=self.scale, fy=self.scale)
            else:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)

            gray_frame = VideoController.gray_conversion(frame)

            if prev_gray_frame is None or self.manager_cv2.key_manager.action or self.manager_cv2.count_frames % 60 == 0:
                self.manager_cv2.key_manager.action = False

                prev_gray_frame = gray_frame
                # This is a method to detect corners
                current_points = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, **self.feature_params)
                mask = np.zeros_like(prev_gray_frame) # Mask image (for drawing)

                continue

            # Calculate OF Lucas Kanade
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, current_points, None, **self.lk_params)

            # Select good points
            good_old = current_points[status == 1]
            good_new = next_points[status == 1]

            # Draw the tracks
            for (new, old) in zip(good_new, good_old):
                old_x, old_y = old.ravel()
                new_x, new_y = new.ravel()

                mask = cv2.line(mask, (new_x, new_y), (old_x, old_y), (100,100,100), 2)
                frame = cv2.circle(frame, (new_x, new_y), 4, (0, 0, 255), -1)

            frame = cv2.add(frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

            final_frame = frame
            if self.concatenate_frame is not None:
                final_frame = self.concatenate_frame(frame, not_processable_frame)

            cv2.imshow(window_name, final_frame)

            # Now update the previous frame and previous points
            prev_gray_frame = gray_frame.copy()
            current_points = good_new.reshape(-1, 1, 2)

            if self.manager_cv2.count_frames % 10 == 0:
                mask = np.zeros_like(prev_gray_frame) # Mask image (for drawing)


class Dense_OF(VideoController):

    def __init__(self, video, stream, fps, scale=1, separate_frame=None, concatenate_frame=None, process_all_frame=True):
        VideoController.__init__(self, video, stream, fps)
        max_corners = 200

        # Parameters for ShiTomasi corner detection
        self.feature_params = {
            'maxCorners': max_corners,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }

        self.scale = scale
        self.separate_frame = separate_frame
        self.concatenate_frame = concatenate_frame
        self.process_all_frame = process_all_frame


    def run(self):
        gray_frame = None
        hsv = None

        for frame in self.manager_cv2:
            if not self.process_all_frame and self.separate_frame is not None:
                frame, not_processable_frame = self.separate_frame(frame, fx=self.scale, fy=self.scale)
            else:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)

            height, width = frame.shape[:2]

            prev_gray_frame = gray_frame
            gray_frame = VideoController.gray_conversion(frame)

            if prev_gray_frame is None or self.manager_cv2.key_manager.action or self.manager_cv2.count_frames % 60 == 0:
                self.manager_cv2.key_manager.action = False

                prev_gray_frame = gray_frame
                hsv = np.zeros((height, width, 3), dtype=np.uint8)
                hsv[..., 1] = 255

                continue

            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 0] = ang * (180 / np.pi / 2)

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('BGR', bgr)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

            frame = cv2.add(frame, bgr*2)

            prev_gray_frame = gray_frame

            frame = self.detect(frame, blob, height, width)
            final_frame = frame
            if not self.process_all_frame and self.concatenate_frame is not None:
                final_frame = self.concatenate_frame(frame, not_processable_frame)
                

            cv2.imshow('Dense_OF', final_frame)


    def detect(self, frame, blob, height, width):
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        selector = SelectorCV2(color=(200,90,0), filled=True, show_vertexes=True)

        for output in layerOutputs:       
            for detection in output:            
                scores = detection[5:]
                classID = np.argmax(scores)           
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height]*2)
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)            
                if len(idxs) > 0:                
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        return selector.draw(frame)


def separate_processable_frame(frame, fx=1, fy=1):
    # Split into processable frame and not processable
    not_processable_frame_height = int(frame.shape[0]/3)
    not_processable_frame = frame[:not_processable_frame_height, :]
    not_processable_frame = cv2.resize(not_processable_frame, None, fx=fx, fy=fy)

    frame = frame[not_processable_frame_height:, :]
    frame = cv2.resize(frame, None, fx=fx, fy=fy)

    return frame, not_processable_frame


def concatenate_processable_frame(frame, not_processable_frame):
    return np.concatenate((not_processable_frame, frame), axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video', default=0,
        help='input video/stream (default 0, it is your main webcam)')

    parser.add_argument('-s', '--stream',
        help='if you pass it, it means that the video is an streaming',
        action='store_true')

    parser.add_argument('-f', '--fps', default=0,
        help='int parameter to indicate the limit of FPS (default 0, it means no limit)',
        type=int)

    parser.add_argument('-a', '--algorithm', default='kanade',
        help='Algorithm to use <kanade>/<dense> Default: kanade')

    parser.add_argument('-S', '--scale', default=1.0,
        help='Scale of the video',
        type=float)
    
    parser.add_argument('-c', '--complete_frame', action='store_true',
                        help='Detect in complete frame (True|False)')

    args = parser.parse_args()

    if type(args.video) is str and args.video.isdigit():
        args.video = int(args.video)

    args.algorithm = args.algorithm.lower()

    optical_detector = {
        'kanade': LucasKanade_OF,
        'dense': Dense_OF,
    }

    if args.algorithm not in optical_detector:
        print('Warning: Algorithm selected invalid. Using default one: kanade')
        args.algorithm = 'kanade'

    kwargs = {}

    if args.scale is not None:
        kwargs['scale'] = args.scale

    of = optical_detector[args.algorithm](args.video, args.stream, args.fps,
            process_all_frame=args.complete_frame,
            separate_frame=separate_processable_frame,
            concatenate_frame=concatenate_processable_frame, **kwargs)
    of.run()