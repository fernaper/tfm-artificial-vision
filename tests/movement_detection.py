# Idea from: https://programarfacil.com/blog/vision-artificial/deteccion-de-movimiento-con-opencv-python/

import cv2

# This is my own library
from cv2_tools.Management import ManagerCV2
from cv2_tools.Selection import SelectorCV2


class MovementDetector():

    
    @staticmethod
    def frame_conversion(frame):
        # 1º Grayscale conversion
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2º Noise suppression
        return cv2.GaussianBlur(gray_frame, (21, 21), 0)


    @staticmethod
    def find_contours(frame, min_area=500):
        contours_img = frame.copy()
        contours, hierarchy = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for i, contour in enumerate(contours):
            # You can check whether a contour with index i is inside another by checking if hierarchy[0,i,3] equals -1 or not.
            # If it is different from -1, then the contour is inside another and we want to ignore it.
            if hierarchy[0, i, 3] != -1:
                continue

            # Ignore small contours
            if cv2.contourArea(contour) < min_area:
                continue
            # Rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            valid_contours.append((x, y, x + w, y + h))

        return valid_contours


    def __init__(self, video, stream, fps, dilate=False):
        self.dilate = dilate
        self.manager_cv2 = ManagerCV2(cv2.VideoCapture(video), is_stream=stream, fps_limit=fps)
        self.manager_cv2.add_keystroke(27, 1, exit=True) # Exit when `Esc`
        self.manager_cv2.add_keystroke(ord(' '), 1, 'action')


    def run(self):
        for frame in self.manager_cv2:
            cv2.imshow("Movement Detector", frame)
        cv2.destroyAllWindows()


class BasicMovementDetector(MovementDetector):

    def __init__(self, video, stream, fps, dilate=True):
        MovementDetector.__init__(self, video, stream, fps, dilate)


    def run(self):
        first_frame = None

        for frame in self.manager_cv2:
            if first_frame is None or self.manager_cv2.key_manager.action:
                self.manager_cv2.key_manager.action = False
                first_frame = MovementDetector.frame_conversion(frame)
                continue

            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            gray_frame = MovementDetector.frame_conversion(frame)
            # 1º Absolute substraction
            substraction = cv2.absdiff(first_frame, gray_frame)
            # 2º Set threshold
            threshold = cv2.threshold(substraction, 25, 255, cv2.THRESH_BINARY)[1]
            # It will help with some holes
            if self.dilate:
                threshold = cv2.dilate(threshold, None, iterations=2)
            # 3º Contour/blob detection
            contours = MovementDetector.find_contours(threshold)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)
            cv2.imshow("BasicMovementDetector", frame)
            cv2.imshow("Threshold", threshold)
            cv2.imshow("Substraction", substraction)

        cv2.destroyAllWindows()


class MOG2MovementDetector(MovementDetector):

    def __init__(self, video, stream, fps, dilate=False):
        MovementDetector.__init__(self, video, stream, fps)


    def run(self):
        back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        for frame in self.manager_cv2:
            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            if self.manager_cv2.new_scene or self.manager_cv2.key_manager.action:
                self.manager_cv2.key_manager.action = False
                print('New scene')
                back_sub.clear()
                back_sub = None
                back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

            fg_mask = back_sub.apply(frame)

            # It will help with some holes
            if self.dilate:
                fg_mask = cv2.dilate(fg_mask, None, iterations=2)

            contours = MovementDetector.find_contours(fg_mask)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)

            cv2.imshow("MOG2MovementDetector", frame)
            cv2.imshow('FG Mask', fg_mask)

        cv2.destroyAllWindows()


class KNNMovementDetector(MovementDetector):

    def __init__(self, video, stream, fps, dilate=False):
        MovementDetector.__init__(self, video, stream, fps, dilate)


    def run(self):
        back_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

        for frame in self.manager_cv2:
            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            if self.manager_cv2.new_scene or self.manager_cv2.key_manager.action:
                self.manager_cv2.key_manager.action = False
                print('New scene')
                back_sub.clear()
                back_sub = None
                back_sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

            fg_mask = back_sub.apply(frame)

            # It will help with some holes
            if self.dilate:
                fg_mask = cv2.dilate(fg_mask, None, iterations=2)

            contours = MovementDetector.find_contours(fg_mask)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)

            cv2.imshow("KNNMovementDetector", frame)
            cv2.imshow('FG Mask', fg_mask)

        cv2.destroyAllWindows()


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

    parser.add_argument('-a', '--algorithm', default='basic',
        help='Algorithm to use <basic>/<mog2>/<knn> Default: basic')

    parser.add_argument('-d', '--dilate', type=bool,
        help='If you want to dilate contours or not (default depends on algorithm)')

    args = parser.parse_args()

    if type(args.video) is str and args.video.isdigit():
        args.video = int(args.video)

    movement_detector = {
        'basic': BasicMovementDetector,
        'mog2': MOG2MovementDetector,
        'knn': KNNMovementDetector
    }

    args.algorithm = args.algorithm.lower()

    if args.algorithm not in movement_detector:
        print('Warning: Algorithm selected invalid. Using default one: basic')
        args.algorithm = 'basic'

    kwargs = {}

    if args.dilate is not None:
        kwargs['dilate'] = args.dilate
        print(kwargs)

    detector_object = movement_detector[args.algorithm](args.video, args.stream, args.fps, **kwargs)
    detector_object.run()