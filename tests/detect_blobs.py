# Standard imports
import cv2
import numpy as np


def draw_rectangle_from_keypoints(frame, keypoints):
    frame = frame.copy()
    for keypoint in keypoints:
        center_x, center_y = keypoint.pt
        size = keypoint.size

        cv2.rectangle(
            frame,
            (int(center_x - size), int(center_y - size)),
            (int(center_x + size), int(center_y + size)),
            (0,255,0),
            2
        )
    return frame


# Read image
im = cv2.imread("blob3.png", cv2.IMREAD_GRAYSCALE)

ret, binary_frame = cv2.threshold(im,15,255,cv2.THRESH_BINARY)

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

im_with_keypoints = draw_rectangle_from_keypoints(im, keypoints)
#im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("Image", im)
cv2.imshow("binary_frame", binary_frame)
cv2.waitKey(0)