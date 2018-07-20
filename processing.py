import cv2
import numpy as np


def Roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drawCorner(img, roi, coord_lst, dimensions, location):
    hgt, wdt, dep = roi.shape
    dst = cv2.cornerHarris(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    goodFeats = cv2.goodFeaturesToTrack(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 1, 0.10, 5)

    if location == 0:
        coordinates = [dimensions[0], dimensions[1], 100]
    elif location == 1:
        coordinates = [dimensions[0], 0, -100]

    if goodFeats is not None:
        if dst.max() < 0.0001:
            cv2.circle(roi, (wdt, hgt), 5, (0, 255, 0), -1)
            coord_lst.append([coordinates[1], coordinates[0]])
        else:
            for circleSet in goodFeats:
                cv2.circle(roi, (circleSet[0][0], circleSet[0][1]), 5, (0, 255, 0), -1)
                coord_lst.append([coordinates[1] - circleSet[0][0] + coordinates[2], coordinates[0]])

    else:
        cv2.circle(roi, (0, hgt), 5, (0, 255, 0), -1)
        coord_lst.append([coordinates[1], coordinates[0]])

    return img, coord_lst


def corners(img):
    dimensions = img.shape  # height, width depth
    wdt = 20
    coord_lst = [(dimensions[1] / 2 - int(dimensions[1] / wdt), int(3 * dimensions[0] / 5)),
                 (dimensions[1] / 2 + int(dimensions[1] / wdt), 3 * dimensions[0] // 5)]

    roi_h = int(dimensions[0] / 10)
    roi_w = int(dimensions[1] / 8)

    bot_right = img[dimensions[0] - roi_h:dimensions[0], dimensions[1] - roi_w:dimensions[1]]

    img, coord_lst = drawCorner(img, bot_right, coord_lst, dimensions, 0)

    bot_left = img[dimensions[0] - roi_h:dimensions[0], 0:roi_w]

    img, coord_lst = drawCorner(img, bot_left, coord_lst, dimensions, 1)

    #print(coord_lst)

    return img, coord_lst


def preprocess2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    minHSV1 = np.array([13, 80, 200])
    maxHSV1 = np.array([30, 255, 255])

    maskHSV1 = cv2.inRange(image, minHSV1, maxHSV1)
    resultHSV1 = cv2.bitwise_and(image, image, mask=maskHSV1)

    minHSV2 = np.array([0, 0, 230])
    maxHSV2 = np.array([180, 20, 255])

    maskHSV2 = cv2.inRange(image, minHSV2, maxHSV2)
    resultHSV2 = cv2.bitwise_and(image, image, mask=maskHSV2)

    final_img = cv2.bitwise_or(resultHSV1, resultHSV2)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2BGR)

    return final_img

def histogram(image):
    dimensions = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)

    mask = np.zeros_like(y)
    vertices = [(dimensions[1] / 2 - int(dimensions[1] / 4), int(3 * dimensions[0] / 5)),
                 (dimensions[1] / 2 + int(dimensions[1] / 4), 3 * dimensions[0] // 5),
                (dimensions[1], dimensions[0]), (0, dimensions[0])]

    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    y = cv2.bitwise_and(y, mask)

    equ = cv2.equalizeHist(y)

    image = cv2.merge([equ,u,v])
    #
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s ,v = cv2.split(image)
    h = cv2.bitwise_and(h,mask)
    final_img = cv2.merge([h,s,v])


    #res = np.hstack((image, final_img))  # stacking images side-by-side

    final_img = cv2.cvtColor(final_img, cv2.COLOR_HSV2BGR)

    #cv2.namedWindow('res', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('res', res)


    return final_img

if __name__ == "__main__":
    img = cv2.imread("test_images/challenge1.png")
    img = histogram(img)
    # img, vertices = corners(img)
    # img = Roi(img, vertices)
    # img = preprocess2(img)
    cv2.imshow("a", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
