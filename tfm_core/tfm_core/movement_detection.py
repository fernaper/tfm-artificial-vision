# Idea from: https://programarfacil.com/blog/vision-artificial/deteccion-de-movimiento-con-opencv-python/

import cv2

from cv2_tools.Selection import SelectorCV2
from tfm_core.utilities import VideoController


class BasicMovementDetector(VideoController):

    def __init__(self, video, stream, fps, dilate=True):
        VideoController.__init__(self, video, stream, fps, dilate)


    def run(self):
        first_frame = None

        for frame in self.manager_cv2:
            if first_frame is None or self.manager_cv2.key_manager.action:
                self.manager_cv2.key_manager.action = False
                first_frame = VideoController.frame_conversion(frame)
                continue

            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            gray_frame = VideoController.frame_conversion(frame)
            # 1º Absolute substraction
            substraction = cv2.absdiff(first_frame, gray_frame)
            # 2º Set threshold
            threshold = cv2.threshold(substraction, 25, 255, cv2.THRESH_BINARY)[1]
            # It will help with some holes
            if self.dilate:
                threshold = cv2.dilate(threshold, None, iterations=2)
            # 3º Contour/blob detection
            contours = VideoController.find_contours(threshold)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)
            cv2.imshow("BasicMovementDetector", frame)
            cv2.imshow("Threshold", threshold)
            cv2.imshow("Substraction", substraction)

        cv2.destroyAllWindows()


class MOG2MovementDetector(VideoController):

    def __init__(self, video, stream, fps, dilate=False, scale=1.0):
        VideoController.__init__(self, video, stream, fps)
        self.__scale = scale


    def run(self):
        back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        for frame in self.manager_cv2:
            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            if self.__scale != 1:
                frame = cv2.resize(frame, None, fx=self.__scale, fy=self.__scale)

            fg_mask, contours = self.next_frame(frame, back_sub)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)

            cv2.imshow("MOG2MovementDetector", frame)
            cv2.imshow('FG Mask', fg_mask)

        cv2.destroyAllWindows()


    def background_substractor(self):
        return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


    def next_frame(self, frame, back_sub):
        if self.manager_cv2.new_scene or self.manager_cv2.key_manager.action:
            self.manager_cv2.key_manager.action = False
            print('New scene')
            back_sub.clear()
            back_sub = None
            back_sub = self.background_substractor()

        fg_mask = back_sub.apply(frame)

        # It will help with some holes
        if self.dilate:
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours = VideoController.find_contours(fg_mask)

        return fg_mask, contours


class KNNMovementDetector(VideoController):

    def __init__(self, video, stream, fps, dilate=False, scale=1.0):
        VideoController.__init__(self, video, stream, fps, dilate)
        self.__scale = scale


    def run(self):
        back_sub = self.background_substractor()

        for frame in self.manager_cv2:
            # Helper selector to draw contours
            selector = SelectorCV2(color=(0, 0, 200), peephole=False)

            if self.__scale != 1:
                frame = cv2.resize(frame, None, fx=self.__scale, fy=self.__scale)

            fg_mask, contours = self.next_frame(frame, back_sub)

            for contour in contours:
                selector.add_zone(contour)

            frame = selector.draw(frame)

            cv2.imshow("KNNMovementDetector", frame)
            cv2.imshow('FG Mask', fg_mask)

        cv2.destroyAllWindows()


    def background_substractor(self):
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)


    def next_frame(self, frame, back_sub):
        if self.manager_cv2.new_scene or self.manager_cv2.key_manager.action:
            self.manager_cv2.key_manager.action = False
            print('New scene')
            back_sub.clear()
            back_sub = None
            back_sub = self.background_substractor()

        fg_mask = back_sub.apply(frame)

        # It will help with some holes
        if self.dilate:
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours = VideoController.find_contours(fg_mask)

        return fg_mask, contours


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video', default=0,
        help='input video/stream (default 0, it is your main webcam)')

    parser.add_argument('-s', '--stream',
        help='if you pass it, it means that the video is an streaming',
        action='store_true')

    parser.add_argument('-S', '--scale', default=1.0,
        help='Scale of the video (default 1.0)',
        type=float)

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

    if args.algorithm != 'basic':
        kwargs['scale'] = args.scale

    if args.dilate is not None:
        kwargs['dilate'] = args.dilate
        print(kwargs)

    detector_object = movement_detector[args.algorithm](args.video, args.stream, args.fps, **kwargs)
    detector_object.run()