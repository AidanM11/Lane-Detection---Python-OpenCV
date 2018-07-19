import cv2
import numpy as np
from preprocessing import preprocessImage
import processing


def detectLines(image, origImg):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 20, 200)
    (height, width) = image.shape
    #cv2.imshow("canny", image)
    #cv2.waitKey()
    imageROI1 = image[:, 0:width//2]
    imageROI2 = image[:, width//2:width]
    #cv2.imshow("ROI", imageROI1)
    #cv2.imshow("ROI", imageROI2)
    #cv2.waitKey()
    lines1 = cv2.HoughLinesP(imageROI1, 6, np.pi/180, 200, lines = 1, minLineLength=100, maxLineGap=400)
    lines2 = cv2.HoughLinesP(imageROI2, 6, np.pi/180, 200, lines = 1, minLineLength=100, maxLineGap=400)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines1 is not None:
        for lineSet in lines1:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.5):
                    cv2.line(origImg, (line[0], line[1]), (line[2], line[3]), (0,0, 255), thickness=10)
    if lines2 is not None:
        for lineSet in lines2:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.5):
                    cv2.line(origImg, (line[0] + width//2, line[1]), (line[2] + width//2, line[3]), (0,0, 255), thickness=10)
    return image

def testLineDetect():
    img = cv2.imread("test_images/solidYellowLeft.jpg")
    img, vertices = processing.corners(img)
    img= processing.Roi(img, vertices)
    img = preprocessImage(img)
    img = detectLines(img)
    cv2.imshow("img", img)
    cv2.waitKey()

def testLineVideo():
    cap = cv2.VideoCapture("test_videos/challenge.mp4")
    while True:
        ret, frame = cap.read()
        origFrame = frame
        frame = preprocessImage(frame)
        frame, vertices = processing.corners(frame)
        frame = processing.Roi(frame, vertices)
        frame = detectLines(frame, origFrame)
        cv2.imshow("Video", origFrame)
        print("frame shown")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

testLineVideo()