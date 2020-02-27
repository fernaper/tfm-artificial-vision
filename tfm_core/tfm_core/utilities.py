import cv2

from cv2_tools.Management import ManagerCV2


class VideoController():

    
    @staticmethod
    def frame_conversion(frame):
        # 1º Grayscale conversion
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2º Noise suppression
        return cv2.GaussianBlur(gray_frame, (21, 21), 0)


    @staticmethod
    def gray_conversion(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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


    @staticmethod
    def get_contours_and_center(frame, min_area=500):
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

            moments = cv2.moments(contour)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            valid_contours.append((x, y, x + w, y + h, (center_x, center_y)))

        return valid_contours


    def __init__(self, video, stream, fps, dilate=False, detect_scenes=False, name='MovementDetector'):
        self.dilate = dilate
        self.name = name
        self.manager_cv2 = ManagerCV2(cv2.VideoCapture(video),
            is_stream=stream, fps_limit=fps, detect_scenes=detect_scenes)
        self.manager_cv2.add_keystroke(27, 1, exit=True) # Exit when `Esc`
        self.manager_cv2.add_keystroke(ord(' '), 1, 'action')


    def run(self):
        for frame in self.manager_cv2:
            cv2.imshow(self.name, frame)
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

    args = parser.parse_args()

    if type(args.video) is str and args.video.isdigit():
        args.video = int(args.video)

    md = VideoController(args.video, args.stream, args.fps)
    md.run()