import numpy as np
import argparse
import cv2
import copy
from edge import EdgeGoban

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True,
        help='Path to the input image.')
    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])
    image = cv2.bilateralFilter(image, 11, 17, 17)

    cv2.drawContours(image, [EdgeGoban(image)], -1, (0, 255, 0), 3)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyWindow("test")
