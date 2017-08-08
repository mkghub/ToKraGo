import numpy as np
import argparse
import cv2
import copy

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True,
	help='Path to the input image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.bilateralFilter(image, 11, 17, 17)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("edged", edged)

_, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in cnts]
max_index = np.argmax(areas)
cnt=cnts[max_index]

cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyWindow("test")
