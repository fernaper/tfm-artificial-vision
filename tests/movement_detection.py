# Idea from: https://programarfacil.com/blog/vision-artificial/deteccion-de-movimiento-con-opencv-python/

import cv2

# This is my own library
from cv2_tools.Management import ManagerCV2
from cv2_tools.Selection import SelectorCV2


def frame_conversion(frame):
    # 1º Grayscale conversion
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2º Noise suppression
    return cv2.GaussianBlur(gray_frame, (21, 21), 0)


def basic_movement_detection(video, stream, fps):
    # Helper manager to get frames from video (+52% faster than basic OpenCV)
    manager_cv2 = ManagerCV2(cv2.VideoCapture(video), is_stream=stream, fps_limit=fps)
    manager_cv2.add_keystroke(27, 1, exit=True) # Exit when `Esc`

    first_frame = None

    for frame in manager_cv2:
        if first_frame is None:
            first_frame = frame_conversion(frame)
            continue

        # Helper selector to draw contours
        selector = SelectorCV2(color=(0, 0, 200), peephole=False)

        gray_frame = frame_conversion(frame)
        # 1º Absolute substraction
        substraction = cv2.absdiff(first_frame, gray_frame)
        # 2º Set threshold
        threshold = cv2.threshold(substraction, 25, 255, cv2.THRESH_BINARY)[1]
        # It will help with some holes
        threshold = cv2.dilate(threshold, None, iterations=2)

        # 3º Contour/blob detection
        contours_img = threshold.copy()
        contours, _ = cv2.findContours(contours_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # Ignore small contours
            if cv2.contourArea(c) < 500:
                continue
            # Rectangle of the contour
            x, y, w, h = cv2.boundingRect(c)
            selection = (x, y, x + w, y + h)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            selector.add_zone(selection)

        frame = selector.draw(frame)
        cv2.imshow("Camera", frame)
        cv2.imshow("Threshold", threshold)
        cv2.imshow("Substraction", substraction)
        cv2.imshow("Contour", contours_img)

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

    basic_movement_detection(args.video, args.stream, args.fps)