'''
YOLO dense clasifier

It is deprecated
'''

import cv2
import numpy as np
import os

from optical_flow import Dense_OF
from tfm_core import config

class Yolo(Dense_OF):

    def __init__(self, net, labels, video, stream, fps, scale=1, confidence = 0.5, threshold = 0.3, **kwargs):
        Dense_OF.__init__(self, video, stream, fps, scale=1, **kwargs)

        np.random.seed(50)
        self.__scale = scale
        self.confidence = confidence
        self.threshold = threshold
        self.net = net

        self.ln = net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        self.labels = labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")


    def run(self):
        gray_frame = None
        hsv = None

        for frame in self.manager_cv2:
            if self.__scale != 1:
                frame = cv2.resize(frame, None, fx=self.__scale, fy=self.__scale)

            gray_frame, hsv, end = self.next_frame(frame, gray_frame, hsv, show=True)

            if end:
                break


    def next_frame(self, frame, gray_frame, hsv, show=False):
        gray_frame, hsv, _ = super().next_frame(frame, gray_frame, hsv)

        thresholding_frame = self.threshold_frame(cv2.split(hsv)[2]) # I am sending the gray layer
        detection_frame = self.detect(thresholding_frame, frame)

        end = False
        if show:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            for region_from, region_to, _ in self.get_regions(thresholding_frame, frame):
                cv2.rectangle(frame, region_from, region_to, (20,0,150), 2)

            cv2.imshow('BGR', bgr)
            cv2.imshow('Thresholding frame', thresholding_frame)
            cv2.imshow('Dense clasifier', detection_frame)
            cv2.imshow('Regions from contours', frame)
            
            if cv2.waitKey(1) == ord('q'):
	            end = True

        return gray_frame, hsv, end


    def threshold_frame(self, gray_frame):
        #TODO: Try other thresholding methods
        _, thresholding_frame = cv2.threshold(gray_frame,15,255,cv2.THRESH_BINARY)
        return thresholding_frame


    def get_regions(self, gray_frame, frame_to_crop, min_dim_size=20):
        contours, _ = cv2.findContours(gray_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w < min_dim_size or h < min_dim_size:
                continue

            yield (x, y), (x+w, y+h), frame_to_crop[y:y+h, x:x+w]


    def detect(self, thresholding_frame, frame):
        frame = frame.copy()
        H, W = frame.shape[:2]

        '''
        regions = self.get_regions(thresholding_frame, frame)

        if not regions:
            return frame'''

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        layerOutputs = self.net.forward(self.ln)

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H]*2)
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y,
                                    int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


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

    parser.add_argument('-S', '--scale', default=1.0,
        help='Scale of the video (default 1.0)',
        type=float)
    
    default_yolo_path = os.path.sep.join([config.PROJECT_PATH, 'tfm-extra'])
    parser.add_argument("-y", "--yolo", default=default_yolo_path,
	    help="Base path to YOLO directory")

    args = parser.parse_args()

    weightsPath = os.path.sep.join([args.yolo, "yolov3.weights"])
    configPath = os.path.sep.join([args.yolo, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    labelsPath = os.path.sep.join([args.yolo, "coco.names"])
    labels = open(labelsPath).read().strip().split("\n")

    kwargs = {}

    if args.scale is not None:
        kwargs['scale'] = args.scale

    dc = Yolo(net, labels, args.video, args.stream, args.fps, **kwargs)
    dc.run()
