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
    lines1 = cv2.HoughLinesP(imageROI1, 4, np.pi/180, 150, lines = 1, minLineLength=100, maxLineGap=600)
    lines2 = cv2.HoughLinesP(imageROI2, 4, np.pi/180, 150, lines = 1, minLineLength=100, maxLineGap=600)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # if lines1 is not None:
    #     for lineSet in lines1:
    #         for line in lineSet:
    #             lineSlope = (line[1] - line[3]) / (line[0] - line[2])
    #             if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
    #                 cv2.line(origImg, (line[0], line[1]), (line[2], line[3]), (0,0, 255), thickness=10)
    # if lines2 is not None:
    #     for lineSet in lines2:
    #         for line in lineSet:
    #             lineSlope = (line[1] - line[3]) / (line[0] - line[2])
    #             if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
    #                 cv2.line(origImg, (line[0] + width//2, line[1]), (line[2] + width//2, line[3]), (0,0, 255), thickness=10)



    #line averaging

    currInd = 0
    points1x = []
    points1y = []
    points2x = []
    points2y = []
    if lines1 is not None:
        for lineSet in lines1:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
                    points1x.append(line[0])
                    points1x.append(line[2])
                    points1y.append(line[1])
                    points1y.append(line[3])
        if len(points1x) > 0:
            m, b = np.polyfit(points1x, points1y, 1)
            for x in range(width//2):
                cv2.circle(origImg, (x, (int)(x*m + b)), 2, (0,0,255))
    if lines2 is not None:
        for lineSet in lines2:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
                    points2x.append(line[0])
                    points2x.append(line[2])
                    points2y.append(line[1])
                    points2y.append(line[3])
        if len(points2x) > 0:
            m, b = np.polyfit(points2x, points2y, 1)
            for x in range(width//2):
                cv2.circle(origImg, (x + width//2, (int)(x*m + b)), 2, (0,0,255))
    return origImg

def testLineDetect():
    img = cv2.imread("test_images/solidYellowLeft.jpg")
    img, vertices = processing.corners(img)
    img= processing.Roi(img, vertices)
    img = preprocessImage(img)
    img = detectLines(img)
    cv2.imshow("img", img)
    cv2.waitKey()

def testLineVideo():
    cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
    while True:
        ret, frame = cap.read()
        origFrame = frame
        frame = preprocessImage(frame)
        frame, vertices = processing.corners(frame)
        frame = processing.Roi(frame, vertices)
        origFrame = detectLines(frame, origFrame)
        cv2.imshow("Video", origFrame)
        print("frame shown")
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imwrite("test_images/newimage.jpg", origFrame)
            break

testLineVideo()