import cv2
import numpy as np

max = 255
cross = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
square = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
xshape = [(0, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]
diamond = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def erode(img, kernel):
    res = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            modify = True
            for (kx, ky) in kernel:
                px = i + kx
                py = j + ky
                if img[px][py] == 0:
                    modify = False

            if modify:
                res[i][j] = max
            else:
                res[i][j] = 0

    return res


def dilate(img, kernel):
    res = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i][j] == max:
                for (kx, ky) in kernel:
                    px = i + kx
                    py = j + ky
                    res[px][py] = max

    return res


def absdiff(img1, img2):
    res = np.zeros((img1.shape[0], img1.shape[1], 1), np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            res[i][j] = abs(img1[i][j] - img2[i][j])
    return res

def overpose(img, corners):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if corners[i][j] == 255:
                # new_img[i][j] = 255
                cv2.circle(new_img, (j, i), 5, (255, 255, 255), thickness=1)
            else:
                new_img[i][j] = img[i][j]

    return new_img

def buildResult(img_path, threshold_val):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img_bin = cv2.threshold(img_gray, threshold_val, max, cv2.THRESH_BINARY_INV)

    R1 = dilate(img_bin, cross)
    R1 = erode(R1, diamond)
    R2 = dilate(img_bin, xshape)
    R2 = erode(R2, square)
    R = absdiff(R1, R2)
    cv2.imshow(img_path, R)

    result = overpose(img_gray, R)
    cv2.imshow("Result for " + img_path, result)


buildResult('./MorpologicalCornerDetection.png', 40)
buildResult('./square-rectangle.png', 200)

while (True):
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()