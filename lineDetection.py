import cv2
import copy
import numpy as np
from preprocessing import preprocessImage
import processing


def detectLines(image, origImg, defaultLine1, defaultLine2, secondPoints):
    allp1 = []
    allp2 = []
    retPoints1x = []
    retPoints1y = []
    retPoints2x = []
    retPoints2y = []
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
    lines1 = cv2.HoughLinesP(imageROI1, 4, np.pi/180, 80, lines = 1, minLineLength=10, maxLineGap=600)
    lines2 = cv2.HoughLinesP(imageROI2, 4, np.pi/180, 80, lines = 1, minLineLength=10, maxLineGap=600)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lineImg = copy.deepcopy(origImg)
    if lines1 is not None:
        for lineSet in lines1:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
                    cv2.line(lineImg, (line[0], line[1]), (line[2], line[3]), (0,0, 255), thickness=10)
    if lines2 is not None:
        for lineSet in lines2:
            for line in lineSet:
                lineSlope = (line[1] - line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.5) or (lineSlope < 100 and lineSlope > 0.3):
                    cv2.line(lineImg, (line[0] + width//2, line[1]), (line[2] + width//2, line[3]), (0,0, 255), thickness=10)



    #line averaging

    currInd = 0
    points1x = []
    points1y = []
    points2x = []
    points2y = []
    polyPoints = []
    bottomPointFound = 10
    lastGoodLine1 = defaultLine1
    print("Last Good:" + (str)(lastGoodLine1))
    lastGoodLine2 = defaultLine2
    print("Last Good2:" + (str)(lastGoodLine2))

    if lines1 is not None:
        for lineSet in lines1:
            for line in lineSet:
                lineSlope = (-line[1] + line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.3) or (lineSlope < 100 and lineSlope > 0.3):
                    yInt = (-line[1]) - (lineSlope * line[0])
                    print("Line Slope:" + str(lineSlope))
                    divisorForYBetweenDots = 100
                    for mult in range(divisorForYBetweenDots):
                        currY = int(height*(mult/divisorForYBetweenDots))
                        p1 = [((-currY-yInt)/lineSlope), currY]
                        if line[0] > line[2]:
                            minX = line[2]
                            maxX = line[0]
                        else:
                            minX = line[0]
                            maxX = line[2]
                        if p1[0] > minX and p1[0] < maxX:
                            allp1.append(p1)
                            #allp2.append(p2)
                            points1x.append(p1[0])
                            #points1x.append(p2[0])
                            points1y.append(p1[1])
                            #points1y.append(p2[1])
        retPoints1x = points1x
        retPoints1y = points1y
        #counting in secondary points
        if secondPoints is not None:
            points1x = points1x + points1x + points1x
            points1y = points1y + points1y + points1y
            for pointSet in secondPoints:
                secondPointsCurr = pointSet[0]
                points1x = points1x + secondPointsCurr[0]
                points1y = points1y + secondPointsCurr[1]

        if len(points1x) > 0:
            a, b, c = np.polyfit(points1y, points1x, 2)
            lastGoodLine1 = [a, b, c]
            allowedLength = (int)(height//3)
            for y in range(height):
                y = height-y
                x = (int)(a*(y*y) + b*y + c)
                cv2.circle(origImg, (x, y), 5, (0,0,255))
                bottomPointFound = bottomPointFound - 1
                #if x < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == 0:
                    polyPoints.append([x, y])
                    bottomPointFound = 10
                if allowedLength == 0:
                    polyPoints.append([x, y])
                    break
    if lastGoodLine1 is not None and len(points1x) <= 0:
        print("Drawing old line")
        a = lastGoodLine1[0]
        b = lastGoodLine1[1]
        c = lastGoodLine1[2]
        allowedLength = (int)(width // 2 * 0.45)
        for y in range(height):
            y = height - y
            x = (int)(a * (y * y) + b * y + c)
            cv2.circle(origImg, (x, y), 5, (0, 0, 255))
            bottomPointFound = bottomPointFound - 1
            if y < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == 0:
                    polyPoints.append([x, y])
                    bottomPointFound = 10
            if allowedLength == 0:
                polyPoints.append([x, y])
                break
    bottomPointFound = 10
    polyPoints = list(reversed(polyPoints))
    if lines2 is not None:
        for lineSet in lines2:
            for line in lineSet:
                lineSlope = (-line[1] + line[3]) / (line[0] - line[2])
                if (lineSlope > -100 and lineSlope < -0.3) or (lineSlope < 100 and lineSlope > 0.3):
                    yInt = (-line[1]) - (lineSlope * line[0])
                    print("Line Slope:" + str(lineSlope))
                    divisorForYBetweenDots = 100
                    for mult in range(divisorForYBetweenDots):
                        currY = int(height*(mult/divisorForYBetweenDots))
                        p1 = [((-currY-yInt)/lineSlope) + width//2, currY]
                        if line[0] > line[2]:
                            minX = line[2]
                            maxX = line[0]
                        else:
                            minX = line[0] + width//2
                            maxX = line[2] + width//2
                        if p1[0] > minX and p1[0] < maxX:
                            allp2.append(p1)
                            #allp2.append(p2)
                            points2x.append(p1[0])
                            #points1x.append(p2[0])
                            points2y.append(p1[1])
                            #points1y.append(p2[1])
        retPoints2x = points2x
        retPoints2y = points2y
        if secondPoints is not None:
            points2x = points2x + points2x + points2x
            points2y = points2y + points2y + points2y
            for pointSet in secondPoints:
                secondPointsCurr = pointSet[1]
                points2x = points2x + secondPointsCurr[0]
                points2y = points2y + secondPointsCurr[1]

        if len(points2x) > 0:
            a, b, c = np.polyfit(points2y, points2x, 2)
            lastGoodLine2 = [a, b, c]
            allowedLength = (int)(height//3)
            for y in range(height):
                y = height-y
                x = (int)(a*(y*y) + b*y + c)
                cv2.circle(origImg, (x, y), 5, (0,0,255))
                bottomPointFound = bottomPointFound - 1
                #if x < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == 0:
                    polyPoints.append([x, y])
                    bottomPointFound = 10
                if allowedLength == 0:
                    print(bottomPointFound)
                    polyPoints.append([x, y])
                    break
    if lastGoodLine1 is not None and len(points1x) <= 0:
        print("Drawing old line")
        a = lastGoodLine1[0]
        b = lastGoodLine1[1]
        c = lastGoodLine1[2]
        allowedLength = (int)(width // 2 * 0.45)
        for y in range(height):
            y = height - y
            x = (int)(a * (y * y) + b * y + c)
            cv2.circle(origImg, (x, y), 5, (0, 0, 255))
            bottomPointFound = bottomPointFound - 1
            if y < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == 0:
                    polyPoints.append([x, y])
                    bottomPointFound = 10
                if allowedLength == 0:
                    polyPoints.append([x, y])
                    break
    if len(polyPoints) > 4:
        # temp = polyPoints[2]
        # polyPoints[2] = polyPoints[3]
        # polyPoints[3] = temp
        polyPoints =np.array(polyPoints)
        newImg = np.zeros_like(origImg)
        cv2.fillPoly(newImg, [polyPoints], (0,255,0))
        origImg = cv2.addWeighted(origImg, 1, newImg, 0.4, 0)
    if len(allp1) > 0:
        for p1 in allp1:
            lineImg = cv2.circle(lineImg, (int(p1[0]), int(p1[1])), 10, (255,0,0), thickness= -1)
    if len(allp2) > 0:
        for p2 in allp2:
            lineImg = cv2.circle(lineImg, (int(p2[0]), int(p2[1])), 10, (255, 0, 0), thickness=-1)
    return origImg, lastGoodLine1, lastGoodLine2, lineImg, [[retPoints1x, retPoints1y], [retPoints2x, retPoints2y]]

def testLineDetect():
    img = cv2.imread("test_images/solidYellowLeft.jpg")
    img, vertices = processing.corners(img)
    img= processing.Roi(img, vertices)
    img = preprocessImage(img)
    img = detectLines(img)
    cv2.imshow("img", img)
    cv2.waitKey()

def testLineVideo():
    cap = cv2.VideoCapture("test_videos/half_road_challenge.mp4")
    lastLine1 = None
    lastLine2 = None
    secondaryPoints = []
    numFramesRelevant = 8
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            origFrame = frame
            frame = processing.preprocess2(frame)
            frame = processing.histogram(frame)
            frame = preprocessImage(frame)
            frame, vertices = processing.corners(frame)
            frame = processing.Roi(frame, vertices)
            if len(secondaryPoints) > numFramesRelevant:
                currFrameSecPoints = secondaryPoints[len(secondaryPoints) - numFramesRelevant:len(secondaryPoints)]
            else:
                currFrameSecPoints = None
            origFrame, lastLine1, lastLine2, lineImg, points = detectLines(frame, origFrame, lastLine1, lastLine2, currFrameSecPoints)
            secondaryPoints.append(points)
            res = np.hstack((origFrame, frame, lineImg))
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video", 1600, 900)
            cv2.imshow("Video", res)
            print("frame shown")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.imwrite("test_images/newimage.jpg", origFrame)
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

testLineVideo()