import cv2
import numpy as np
import os

from tfm_core.optical_flow import Dense_OF
from tfm_core.dnn import utilities


class DenseClasifier(Dense_OF):

    def __init__(self, labels, video, stream, fps, confidence = 0.5, scale=1, model='resnet', **kwargs):
        Dense_OF.__init__(self, video, stream, fps, scale=1, **kwargs)

        np.random.seed(50)
        self.__scale = scale
        self.confidence = confidence
        self.model = model

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
        detected_regions = self.detect(thresholding_frame, frame)

        end = False
        if show:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            for region_from, region_to, detected_class, confidence in detected_regions:
                if self.labels[detected_class] == 'background':
                    continue

                color = [int(c) for c in self.colors[detected_class]]

                cv2.rectangle(frame, region_from, region_to, color, 2)

                cv2.putText(frame, '{} ({:.2f})'.format(self.labels[detected_class], confidence),
                    (region_from[0], region_from[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            cv2.imshow('BGR', bgr)
            cv2.imshow('Thresholding frame', thresholding_frame)
            cv2.imshow('DNN', frame)
            
            if cv2.waitKey(1) == ord('q'):
	            end = True

        return gray_frame, hsv, end


    def threshold_frame(self, gray_frame, use_otsu=True):
        threshold_method = cv2.THRESH_BINARY

        if use_otsu:
            threshold_method += cv2.THRESH_OTSU

        _, thresholding_frame = cv2.threshold(gray_frame, 17, 255, threshold_method)

        return thresholding_frame


    def get_regions(self, gray_frame, frame_to_crop, min_dim_size=20):
        height, width = gray_frame.shape[:2]
        contours, _ = cv2.findContours(gray_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)

            # If it is a really small region
            if w < min_dim_size or h < min_dim_size:
                continue

            # If it is a really big region
            if h > height / 2 or w > width / 2:
                continue

            yield (x, y), (x+w, y+h), frame_to_crop[y:y+h, x:x+w]


    def detect(self, thresholding_frame, frame):
        for region_from, region_to, cropped_frame in self.get_regions(thresholding_frame, frame):
            cropped_frame = cv2.resize(cropped_frame, (64,64))

            predictions = utilities.send_frame_serving_tf(cropped_frame, model=self.model)
            detected_class_index = np.where(predictions == np.amax(predictions))[0][0]

            if max(predictions) < self.confidence:
                continue

            # Touples (region_from, region_to, class_id, confidence)
            yield region_from, region_to, detected_class_index, max(predictions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video', default=0,
        help='input video/stream (default 0, it is your main webcam)')

    parser.add_argument('-d', '--dataset', default='dataset',
        help='dataset you want to use (default dataset)')

    parser.add_argument('-s', '--stream',
        help='if you pass it, it means that the video is an streaming',
        action='store_true')

    parser.add_argument('-f', '--fps', default=0,
        help='int parameter to indicate the limit of FPS (default 0, it means no limit)',
        type=int)

    parser.add_argument('-S', '--scale', default=1.0,
        help='Scale of the video (default 1.0)',
        type=float)

    parser.add_argument('-m', '--model', default='resnet',
        help='model name to call (default resnet)')

    args = parser.parse_args()
    kwargs = {}

    if args.scale is not None:
        kwargs['scale'] = args.scale

    dc = DenseClasifier(utilities.get_labels(args.dataset), args.video, args.stream, args.fps, model=args.model, **kwargs)
    dc.run()
