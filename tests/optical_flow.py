# Idea from: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# http://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/Lucas-Kanade2.pdf
# https://www.youtube.com/watch?v=1r8E9uAcn4E

import cv2
import numpy as np

from utilities import VideoController

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

    def __init__(self, video, stream, fps, **kwargs):
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

    
    def run(self):
        gray_frame = None
        current_points = None
        mask = None

        for frame in self.manager_cv2:
            prev_gray_frame = gray_frame
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
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                old_x, old_y = old.ravel()
                new_x, new_y = new.ravel()

                mask = cv2.line(mask, (new_x, new_y), (old_x, old_y), (100,100,100), 2)
                frame = cv2.circle(frame, (new_x, new_y), 4, (0, 0, 255), -1)

            frame = cv2.add(frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

            cv2.imshow('Lucas Kanade', frame)

            # Now update the previous frame and previous points
            prev_gray_frame = gray_frame.copy()
            current_points = good_new.reshape(-1, 1, 2)

            if self.manager_cv2.count_frames % 10 == 0:
                mask = np.zeros_like(prev_gray_frame) # Mask image (for drawing)


class Dense_OF(VideoController):

    def __init__(self, video, stream, fps, scale=1):
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

    
    def run(self):
        gray_frame = None
        hsv = None

        for frame in self.manager_cv2:
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
            frame = cv2.add(frame, bgr*2)

            prev_gray_frame = gray_frame

            cv2.imshow('Dense_OF', frame)


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

    of = optical_detector[args.algorithm](args.video, args.stream, args.fps, **kwargs)
    of.run()