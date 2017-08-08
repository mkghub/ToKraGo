"""Automatic Go/Baduk/Weiqi recorder."""
import argparse
import cv2
import numpy as np
from edge import CountourGoban, HoughGoban

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Path to the input image.')
    args = vars(ap.parse_args())

    image = cv2.imread(args['image'])
    image = cv2.bilateralFilter(image, 11, 17, 17)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150,apertureSize = 3)

    #CountourGoban(edged, image)
    HoughGoban(edged, image)
