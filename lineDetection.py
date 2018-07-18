import cv2
import numpy as np
from preprocessing import preprocessImage


def detectLines(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 20, 200)
    (height, width) = image.shape
    cv2.imshow("canny", image)
    cv2.waitKey()
    imageROI1 = image[0:width//2, :]
    imageROI2 = image[width//2:width, :]
    lines1 = cv2.HoughLinesP(imageROI1, 6, np.pi/180, 400, lines = 1, minLineLength=100, maxLineGap=320)
    lines2 = cv2.HoughLinesP(imageROI2, 6, np.pi/180, 400, lines = 1, minLineLength=100, maxLineGap=320)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for lineSet in lines1:
        for line in lineSet:
            lineSlope = (line[1] - line[3]) / (line[0] - line[2])
            if (lineSlope > -8 and lineSlope < -0.6) or (lineSlope < 8 and lineSlope > 0.6):
                cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0,0, 255), thickness=10)
    for lineSet in lines2:
        for line in lineSet:
            lineSlope = (line[1] - line[3]) / (line[0] - line[2])
            if (lineSlope > -8 and lineSlope < -0.6) or (lineSlope < 8 and lineSlope > 0.6):
                cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0,0, 255), thickness=10)
    return image

img = cv2.imread("test_images/solidYellowCurve.jpg")
img = preprocessImage(img)
img = detectLines(img)
cv2.imshow("img", img)
cv2.waitKey()