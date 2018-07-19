import cv2
import numpy as np
import os

def Roi(img, vertices):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def corners(img):
    height, width, depth = img.shape
    coord_lst = [(width / 2, height / 2)]

    bot_right = img[height - 100:height, width - 100:width]

    dst = cv2.cornerHarris(cv2.cvtColor(bot_right, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    hgt, wdt, dep = bot_right.shape
    goodFeats = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_right, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)
    if goodFeats is not None:
        if dst.max() < 0.0001:
            cv2.circle(bot_right, (wid, hgt), 3, (0, 0, 255), -1)
            coord_lst.append([width, height])
        else:
            for circleSet in goodFeats:
                cv2.circle(bot_right, (circleSet[0][0], circleSet[0][1]), 3, (0, 0, 255), -1)
                coord_lst.append([width - circleSet[0][0],height])

    else:
        cv2.circle(bot_right, (0, hgt), 3, (0, 0, 255), -1)
        coord_lst.append([width, height])




    bot_left = img[height - 100:, :100]

    dst = cv2.cornerHarris(cv2.cvtColor(bot_left, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    hgt, wdt, dep = bot_left.shape
    goodFeats1 = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_left, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)
    if goodFeats1 is not None:
        if dst.max() < 0.0001:
            cv2.circle(bot_left, (0, hgt), 3, (0, 0, 255), -1)
            coord_lst.append([0, height])
        else:
            for circleSet in goodFeats1:
                cv2.circle(bot_left, (circleSet[0][0], circleSet[0][1]), 3, (0, 0, 255), -1)
                coord_lst.append([circleSet[0][0],height])

    else:
        cv2.circle(bot_left, (0, hgt), 3, (0, 0, 255), -1)
        coord_lst.append([0, height])

    return img, coord_lst


#two_up = os.path.abspath(os.path.join(__file__, "../../test_images/solidWhiteCurve.jpg"))
# img = cv2.imread("test_images/solidYellowCurve2.jpg")
# cv2.imshow("orig", img)
#
# print(img.shape)
#
# corner, vertices = corners(img)
#
# im1 = Roi(img, vertices)
#
# cv2.imshow("Image", im1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
