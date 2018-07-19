import cv2
import numpy as np
from processing import Roi,corners

def preprocessImage(image):
    (height, width, depth) = image.shape
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #mask = np.zeros((height, width, 1), np.uint8)
    midHgt = height // 2
    #mask[midHgt:height, :, :] = 255
    #mask = cv2.merge((mask, mask, mask))
    print(image.shape)
    #maskedImage = cv2.bitwise_and(image, mask)
    gImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshedMask = cv2.threshold(gImage, 180, 255, cv2.THRESH_BINARY)
    print(threshedMask.shape)
    threshedMask = cv2.merge((threshedMask, threshedMask, threshedMask))
    threshImage = cv2.bitwise_and(image, threshedMask)
    threshImage = cv2.GaussianBlur(threshImage, (5,5), 0)
    kern1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    #threshImage = cv2.morphologyEx(threshImage, cv2.MORPH_CLOSE, kern1)
    return threshImage


if __name__ == '__main__':
    img = cv2.imread("test_images/solidYellowLeft.jpg")

    img, vertices = corners(img)
    img = Roi(img, vertices)

    print(vertices)

    cv2.imshow("processed",preprocessImage(img))

    cap = cv2.VideoCapture('test_videos/challenge.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame, vertices = corners(frame)
            frame = Roi(frame, vertices)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if cv2.waitKey(25) & 0xFF == ord(' '):
                cv2.imwrite("image1.jpg", frame)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()