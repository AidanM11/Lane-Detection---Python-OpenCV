import cv2
import numpy as np
import os


def process(img):
    return


def Roi(img):
    height, width, depth = img.shape
    vertices = [(0, height), (0, height / 2), (width / 2, height / 2), (width, height)]

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def corners(img):
    coord_lst = []
    height, width, depth = img.shape

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bot_right = img[height - 100:height, width - 100:width]

    goodFeats = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_right, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)

    for circleSet in goodFeats:
        for circle in circleSet:
            print(circle[0])
            cv2.circle(bot_right, (int(circle[0]), int(circle[1])), 2, (0, 0, 255), -1)

    bot_left = img[height - 100:, :100]
    goodFeats1 = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_left, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)
    # for circleSet in goodFeats1:
    #     if type(circleSet) == None:
    #         cv2.circle(bot_left, (0, height), 2, (0, 0, 255), -1)
    #     else:
    #         for circle in circleSet:
    #             cv2.circle(bot_left, (circle[0], circle[1]), 2, (0, 0, 255), -1)
    #             print(circle)

    dst = cv2.cornerHarris(cv2.cvtColor(bot_right,cv2.COLOR_BGR2GRAY),2,3,0.04)
    dst = cv2.dilate(dst,None)
    print(dst.max()) #must be greater than 0.001 to be considered a corner
    cv2.imshow("eigen",bot_left)

    return img


two_up = os.path.abspath(os.path.join(__file__, "../../test_images/solidWhiteCurve.jpg"))
img = cv2.imread(two_up)
print(img.shape)

corner = corners(img)

cv2.imshow("Image", corner)
cv2.waitKey(0)
cv2.destroyAllWindows()
