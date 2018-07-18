import cv2
import numpy as np


def process(img):
    return


def Roi(img):
    height, width, depth = img.shape
    vertices = [(0, height), (0,height/2), (width / 2, height / 2), (width, height)]

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def corners(img):
    height, width, depth = img.shape

    img = img[height-100:height,width-100:height,:]
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    goodFeats = cv2.goodFeaturesToTrack(grayImg, 300, 0.01, 5)

    for circleSet in goodFeats:
        for circle in circleSet:
            cv2.circle(img1, (circle[0], circle[1]), 2, (0, 0, 255), -1)
    return

img = cv2.imread("/Accounts/linr2/Desktop/Test/test_images/solidWhiteCurve.jpg")
print(img.shape)

region = Roi(img)


cv2.imshow("Image", region)
cv2.waitKey(0)
cv2.destroyAllWindows()

