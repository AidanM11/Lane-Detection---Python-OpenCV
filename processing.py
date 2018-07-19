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
    print(img.shape)
    coord_lst = [(width / 2 + int(width/10), height / 2), (width / 2 - int(width/10), height / 2)]

    roi_h = int(height/10)
    roi_w = int(width/8)

    bot_right = img[height - roi_h:height, width - roi_w:width]

    dst = cv2.cornerHarris(cv2.cvtColor(bot_right, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    hgt, wdt, dep = bot_right.shape
    goodFeats = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_right, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)
    if goodFeats is not None:
        if dst.max() < 0.0001:
            cv2.circle(bot_right, (wdt, hgt), 5, (0, 255, 0), -1)
            coord_lst.append([width, height])
        else:
            for circleSet in goodFeats:
                cv2.circle(bot_right, (circleSet[0][0], circleSet[0][1]), 5, (0, 255, 0), -1)
                coord_lst.append([width - circleSet[0][0]+100,height])

    else:
        cv2.circle(bot_right, (0, hgt), 5, (0, 255, 0), -1)
        coord_lst.append([width, height])




    bot_left = img[height - roi_h:height, 0:roi_w]

    dst = cv2.cornerHarris(cv2.cvtColor(bot_left, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    hgt, wdt, dep = bot_left.shape
    goodFeats1 = cv2.goodFeaturesToTrack(cv2.cvtColor(bot_left, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)
    if goodFeats1 is not None:
        if dst.max() < 0.0001:
            cv2.circle(bot_left, (0, hgt), 5, (0, 255, 0), -1)
            coord_lst.append([0, height])
        else:
            for circleSet in goodFeats1:
                cv2.circle(bot_left, (circleSet[0][0], circleSet[0][1]), 5, (0, 255, 0), -1)
                coord_lst.append([circleSet[0][0]-200,height])

    else:
        cv2.circle(bot_left, (0, hgt), 5, (0, 255, 0), -1)
        coord_lst.append([0, height])

    return img, coord_lst

def preprocess2(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    thresh = 80
    thresh2 = 55

    bgr = [[0,255,255],[255,255,255]]
    hsv1 = cv2.cvtColor(np.uint8([[bgr[0]]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[bgr[1]]]), cv2.COLOR_BGR2HSV)[0][0]

    maxHSV1 = np.array([hsv1[0] + thresh, hsv1[1] + 150, hsv1[2] + 150])
    minHSV1 = np.array([hsv1[0] - thresh, hsv1[1] - 150, hsv1[2] - 150])

    maskHSV1 = cv2.inRange(image, minHSV1, maxHSV1)
    resultHSV1 = cv2.bitwise_and(image, image, mask=maskHSV1)

    maxHSV2 = np.array([hsv2[0]+150, hsv2[1]+150, hsv2[2]])
    minHSV2 = np.array([hsv2[0]-150, hsv2[1]-150, hsv2[2] - thresh2])

    maskHSV2 = cv2.inRange(image, minHSV2, maxHSV2)
    resultHSV2 = cv2.bitwise_and(image, image, mask=maskHSV2)

    final_img = cv2.bitwise_or(resultHSV1,resultHSV2)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2BGR)


    return final_img


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
