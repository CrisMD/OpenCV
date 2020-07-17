import cv2
import platform
print(cv2.__version__)
print(platform.architecture())

img = cv2.imread('D:/0.png')

cv2.imshow("image",img)
cv2.moveWindow("image",200,200)
cv2.waitKey(0)

