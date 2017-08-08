"""Contain functions needed for contour goban detection."""
import cv2
import numpy as np


def CountourGoban(edged, image):
    _, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt = cnts[max_index]

    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyWindow("test")

def HoughGoban(edged,image):
    lines = cv2.HoughLines(edged,1,np.pi/180,180)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("test", image)
    cv2.imshow("edged", edged)
    cv2.waitKey(0)
    cv2.destroyWindow("test")


def ShowImage(images, imageWindowNames):
    for image, imageWindowName in zip(images, imageWindowNames):
        cv2.imshow(imageWindowName, image)
    cv2.waitKey(0)
    cv2.destroyWindow(imageWindowName)
