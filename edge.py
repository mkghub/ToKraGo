"""Contain functions needed for contour goban detection."""
import cv2
import numpy as np


def EdgeGoban(image):
    """Return contour of detected goban."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow("edged", edged)

    _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt = cnts[max_index]

    return(cnt)
