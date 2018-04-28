import cv2
import numpy as np


def contour_goban(edged, image):
    _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt = cnts[max_index]

    image_cnt = cv2.drawContours(image.copy(), [cnt], -1, (0, 255, 0), 3)
    return image_cnt


def hough_goban(edged, image):
    lines = cv2.HoughLines(edged, 1, np.pi/180, 180)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

def get_image_edges(image, binary_threshold=(0, 255), canny_threshold=(0, 200)):
    _, thresh = cv2.threshold(image, binary_threshold[0], binary_threshold[1], 0)
    thresh_blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    edges = cv2.Canny(thresh_blurred, canny_threshold[0], canny_threshold[1])

    return edges


