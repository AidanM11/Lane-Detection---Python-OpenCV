import cv2
import copy
import numpy as np
from preprocessing import preprocessImage
import processing


def detectLines(image, origImg, defaultLine1, defaultLine2):
    allp1 = []
    allp2 = []
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
    lines1 = cv2.HoughLinesP(imageROI1, 4, np.pi/180, 120, lines = 1, minLineLength=100, maxLineGap=600)
    lines2 = cv2.HoughLinesP(imageROI2, 4, np.pi/180, 120, lines = 1, minLineLength=100, maxLineGap=600)
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
    bottomPointFound = False
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
                        p1 = [(-currY-yInt)/lineSlope, currY]
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
        if len(points1x) > 0:
            a, b, c = np.polyfit(points1y, points1x, 2)
            lastGoodLine1 = [a, b, c]
            allowedLength = (int)(height//3)
            for y in range(height):
                y = height-y
                x = (int)(a*(y*y) + b*y + c)
                cv2.circle(origImg, (x, y), 5, (0,0,255))
                #if x < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == False:
                    polyPoints.append([x, y])
                    bottomPointFound = True
                if allowedLength == 0:
                    polyPoints.append([x, y])
                    break
    if lastGoodLine1 is not None and len(points1x) <= 0:
        print("Drawing old line")
        m = lastGoodLine1[0]
        b = lastGoodLine1[1]
        allowedLength = (int)(width // 2 * 0.45)
        for x in range(width // 2):
            cv2.circle(origImg, (x, (int)(x * m + b)), 5, (0, 0, 255))
            if (int)(x * m + b) < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == False:
                    polyPoints.append([x, (int)(x * m + b)])
                    bottomPointFound = True
            if allowedLength == 0:
                polyPoints.append([x, (int)(x * m + b)])
                break
    bottomPointFound = False
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
            lastGoodLine2 = [m, b]
            allowedLength = (int)(width // 2 * 0.45)
            for x in range(width//2):
                x = width - x - width//2
                cv2.circle(origImg, (x + width//2, (int)(x*m + b)), 5, (0,0,255))
                if (int)(x*m + b) < height:
                    allowedLength = allowedLength - 1
                    if bottomPointFound == False:
                        polyPoints.append([x + width//2, (int)(x*m + b)])
                        bottomPointFound = True
                if allowedLength == 0:
                    polyPoints.append([x + width//2, (int)(x * m + b)])
                    break
    if lastGoodLine2 is not None and len(points2x) <= 0:
        print("Drawing old line")
        m = lastGoodLine2[0]
        b = lastGoodLine2[1]
        allowedLength = (int)(width // 2 * 0.45)
        for x in range(width // 2):
            x = width - x - width // 2
            cv2.circle(origImg, (x + width // 2, (int)(x * m + b)), 5, (0, 0, 255))
            if (int)(x * m + b) < height:
                allowedLength = allowedLength - 1
                if bottomPointFound == False:
                    polyPoints.append([x + width // 2, (int)(x * m + b)])
                    bottomPointFound = True
            if allowedLength == 0:
                polyPoints.append([x + width // 2, (int)(x * m + b)])
                break
    if len(polyPoints) == 4:
        temp = polyPoints[2]
        polyPoints[2] = polyPoints[3]
        polyPoints[3] = temp
        polyPoints =np.array(polyPoints)
        newImg = np.zeros_like(origImg)
        cv2.fillPoly(newImg, [polyPoints], (0,255,0))
        origImg = cv2.addWeighted(origImg, 1, newImg, 0.7, 0)
    if len(allp1) > 0:
        for p1 in allp1:
            lineImg = cv2.circle(lineImg, (int(p1[0]), int(p1[1])), 10, (255,0,0), thickness= -1)
    if len(allp2) > 0:
        for p2 in allp2:
            lineImg = cv2.circle(lineImg, (int(p2[0]), int(p2[1])), 10, (255, 0, 0), thickness=-1)
    return origImg, lastGoodLine1, lastGoodLine2, lineImg

def testLineDetect():
    img = cv2.imread("test_images/solidYellowLeft.jpg")
    img, vertices = processing.corners(img)
    img= processing.Roi(img, vertices)
    img = preprocessImage(img)
    img = detectLines(img)
    cv2.imshow("img", img)
    cv2.waitKey()

def testLineVideo():
    cap = cv2.VideoCapture("test_videos/project_video.mp4")
    lastLine2 = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            origFrame = frame
            frame = processing.preprocess2(frame, 1)
            frame = processing.histogram(frame)
            frame = preprocessImage(frame)
            frame, vertices = processing.corners(frame)
            frame = processing.Roi(frame, vertices)
            origFrame, lastLine1, lastLine2, lineImg = detectLines(frame, origFrame, lastLine1, lastLine2)
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