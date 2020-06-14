import cv2
import numpy as np
import os

from tfm_core.optical_flow import Dense_OF
from tfm_core.movement_detection import MOG2MovementDetector, KNNMovementDetector
from tfm_core.dnn import utilities


class DenseClassifier(Dense_OF):

    def __init__(self, labels, video, stream, fps, confidence = 0.5, scale=1, model='resnet', dnn_size=64, **kwargs):
        Dense_OF.__init__(self, video, stream, fps, scale=1, **kwargs)

        np.random.seed(50)
        self.__scale = scale
        self.confidence = confidence
        self.model = model

        self.labels = labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
        self.dnn_size = dnn_size
        self.measure_performance = kwargs.get('measure_performance', False)
        self.draw_frame = kwargs.get('draw_frame', True)


    def run(self):
        gray_frame = None
        hsv = None

        for frame in self.manager_cv2:
            if self.__scale != 1:
                frame = cv2.resize(frame, None, fx=self.__scale, fy=self.__scale)

            gray_frame, hsv, end = self.next_frame(frame, gray_frame, hsv, show=True)

            if end:
                break

        if self.measure_performance:
            print(self.manager_cv2.get_fps())


    def next_frame(self, frame, gray_frame, hsv, show=False):
        gray_frame, hsv, _ = super().next_frame(frame, gray_frame, hsv)

        thresholding_frame = self.threshold_frame(cv2.split(hsv)[2]) # I am sending the gray layer
        detected_regions = self.detect(thresholding_frame, frame)

        end = False
        if show:
            bgr = None
            if self.draw_frame:
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            for region_from, region_to, detected_class, confidence in detected_regions:
                if not self.draw_frame:
                    # Consumes the output
                    continue

                if self.labels[detected_class] == 'background':
                    continue

                color = [int(c) for c in self.colors[detected_class]]

                cv2.rectangle(frame, region_from, region_to, color, 2)

                cv2.putText(frame, '{} ({:.2f})'.format(self.labels[detected_class], confidence),
                    (region_from[0], region_from[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            if self.draw_frame:
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
            cropped_frame = cv2.resize(cropped_frame, (self.dnn_size,self.dnn_size))

            predictions = utilities.send_frame_serving_tf(cropped_frame, model=self.model)

            if predictions is None:
                continue

            detected_class_index = np.where(predictions == np.amax(predictions))[0][0]

            if max(predictions) < self.confidence:
                continue

            # Touples (region_from, region_to, class_id, confidence)
            yield region_from, region_to, detected_class_index, max(predictions)


class MovementClassifier(MOG2MovementDetector, KNNMovementDetector):

    def __init__(self, labels, video, stream, fps, confidence = 0.5, scale=1, model='resnet', dnn_size=64, parent=None, **kwargs):
        super().__init__(video, stream, fps, scale=1, **kwargs)

        np.random.seed(50)
        self.__scale = scale
        self.confidence = confidence
        self.model = model

        self.labels = labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
        self.dnn_size = dnn_size

        self.parent = parent
        self.measure_performance = kwargs.get('measure_performance', False)
        self.draw_frame = kwargs.get('draw_frame', True)


    def run(self):
        back_sub = self.parent.background_substractor(self)

        for frame in self.manager_cv2:
            if self.__scale != 1:
                frame = cv2.resize(frame, None, fx=self.__scale, fy=self.__scale)

            fg_mask, frame, end = self.next_frame(frame, back_sub, show=True)

            if self.draw_frame:
                cv2.imshow('FG Mask', fg_mask)
                cv2.imshow('DNN', frame)

            if end:
                break

        if self.measure_performance:
            print(self.manager_cv2.get_fps())

        cv2.destroyAllWindows()


    def next_frame(self, frame, back_sub, show=False):
        if self.parent == None:
            print('Must select parent')
            return None, None, True

        fg_mask, contours = self.parent.next_frame(self, frame, back_sub)
        detected_regions = self.detect(contours, frame)

        end = False

        if show:
            for region_from, region_to, detected_class, confidence in detected_regions:
                if self.labels[detected_class] == 'background':
                    continue

                if not self.draw_frame:
                    # Consumes the output
                    continue

                color = [int(c) for c in self.colors[detected_class]]

                cv2.rectangle(frame, region_from, region_to, color, 2)

                cv2.putText(frame, '{} ({:.2f})'.format(self.labels[detected_class], confidence),
                    (region_from[0], region_from[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            if self.draw_frame and cv2.waitKey(1) == ord('q'):
                end = True

        return fg_mask, frame, end


    def get_regions(self, contours, frame_to_crop, min_dim_size=20):
        height, width = frame_to_crop.shape[:2]

        for x1, y1, x2, y2 in contours:
            w = x2 - x1
            h = y2 - y1

            # If it is a really small region
            if w < min_dim_size or h < min_dim_size:
                continue

            # If it is a really big region
            if h > height / 2 or w > width / 2:
                continue

            yield (x1, y1), (x2, y2), frame_to_crop[y1:y2, x1:x2]


    def detect(self, contours, frame):
        for region_from, region_to, cropped_frame in self.get_regions(contours, frame):
            cropped_frame = cv2.resize(cropped_frame, (self.dnn_size,self.dnn_size))

            predictions = utilities.send_frame_serving_tf(cropped_frame, model=self.model)

            if predictions is None:
                continue

            detected_class_index = np.where(predictions == np.amax(predictions))[0][0]

            if max(predictions) < self.confidence:
                continue

            # Touples (region_from, region_to, class_id, confidence)
            yield region_from, region_to, detected_class_index, max(predictions)


if __name__ == "__main__":
    # python3 detect.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d medium_mio-tcd_dataset_9 -m resnet -i 64 -a knn
    # python3 detect.py -S 0.5 -v ../../videos/20181006_130650.mp4 -d medium_mio-tcd_dataset_9 -m alexnet -i 227 -a knn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video', default=0,
        help='input video/stream (default 0, it is your main webcam)')

    parser.add_argument('-a', '--algorithm', default='dense',
        help='object detector algorithm <dense>/<mog2>/<knn> (default: dense)')

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

    parser.add_argument('-i', '--image_size', default=64,
        help='Size of the image sended to the neural network (default 64)',
        type=int)

    parser.add_argument('-n', '--no_interface', action='store_false',
                        help='Show interface (True|False)')

    parser.add_argument('-M', '--measure_performance', action='store_true',
                        help='Measure performance (True|False)')

    args = parser.parse_args()
    kwargs = {}

    if args.scale is not None:
        kwargs['scale'] = args.scale

    args.algorithm = args.algorithm.lower()

    algorithms = {
        'dense': DenseClassifier,
        'mog2': MovementClassifier,
        'knn': MovementClassifier
    }

    if args.algorithm not in algorithms:
        print('Warning: Algorithm selected invalid. Using default one: dense')
        args.algorithm = 'dense'

    if args.algorithm != 'dense':
        parents = {
            'mog2': MOG2MovementDetector,
            'knn': KNNMovementDetector
        }

        kwargs['parent'] = parents[args.algorithm]

    if not args.no_interface:
        kwargs['draw_frame'] = False

    kwargs['measure_performance'] = args.measure_performance

    dc = algorithms[args.algorithm](utilities.get_labels(args.dataset), args.video, args.stream, args.fps, model=args.model, dnn_size=args.image_size, **kwargs)
    dc.run()
