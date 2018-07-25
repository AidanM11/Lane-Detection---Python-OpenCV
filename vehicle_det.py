import cv2
import numpy as np
import os


'''
Resources:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
https://github.com/ckirksey3/vehicle-detection-with-svm
'''

#64 x 64 images



# def hog(img):
#     bin_n = 16
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     mag, ang = cv2.cartToPolar(gx, gy)
#
#     # quantizing binvalues in (0...16)
#     bins = np.int32(bin_n*ang/(2*np.pi))
#
#     # Divide to 4 sub-squares
#     bin_cells = bins[:32,:32], bins[32:,:32], bins[:32,32:], bins[32:,32:]
#     mag_cells = mag[:32,:32], mag[32:,:32], mag[:32,32:], mag[32:,32:]
#     hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#     hist = np.hstack(hists)
#     return hist

img = cv2.imread("vehicles/GTI_MiddleClose/image0000.png")

imgNames = []

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))


hog = cv2.HOGDescriptor()

responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

samples = []


print(responses.size)

for root, dirs, files in os.walk("/Accounts/linr2/Desktop/Test/vehicles/GTI_Far"):
    for filename in files:
        if filename.endswith(".png") and not filename.startswith('._'):
            imgNames.append(os.path.join(root,filename))

labels = np.repeat(np.arange(10),len(imgNames)//10+1)

for i in imgNames:
    img = cv2.imread(i)
    h = hog.compute(img)

    samples.append(h)
    samples = np.float32(samples)

svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)



#h = hog.compute(img)



cv2.waitKey(0)
cv2.destroyAllWindows()