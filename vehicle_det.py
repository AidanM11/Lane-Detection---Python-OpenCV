import cv2
import numpy as np


'''
Resources:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
https://github.com/ckirksey3/vehicle-detection-with-svm
'''

def hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

#img = cv2.imread("vehicles/GTI_MiddleClose/._image0000.png")
img = cv2.imread("test_images/solidYellowCurve.jpg")

#
img = hog(img)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()