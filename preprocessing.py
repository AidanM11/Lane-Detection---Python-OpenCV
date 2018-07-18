import cv2
import numpy as np

def preprocessImage(image):
    (height, width, depth) = image.shape
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((height, width, 1), np.uint8)
    midHgt = height // 2
    mask[midHgt:height, :, :] = 255
    mask = cv2.merge((mask, mask, mask))
    print(image.shape)
    maskedImage = cv2.bitwise_and(image, mask)
    gImage = cv2.cvtColor(maskedImage, cv2.COLOR_BGR2GRAY)
    ret, threshedMask = cv2.threshold(gImage, 180, 255, cv2.THRESH_BINARY)
    print(threshedMask.shape)
    threshedMask = cv2.merge((threshedMask, threshedMask, threshedMask))
    threshImage = cv2.bitwise_and(image, threshedMask)
    return threshImage



img = cv2.imread("test_images/solidYellowCurve2.jpg")
cv2.imshow("orig", img)
cv2.imshow("processed",preprocessImage(img))
cv2.waitKey()
