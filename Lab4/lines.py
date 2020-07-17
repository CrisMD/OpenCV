import cv2
import numpy as np

img = cv2.imread('sudoku.png')
img = cv2.resize(img, (512, 512))
cv2.imshow("Original", img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Detect the edges of the image by using a Canny detector
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
cv2.imshow("Edges", edges)

# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter \theta in radians. We use 1 degree (np.pi/180)
# threshold: The minimum number of intersections to “detect” a line
lines = cv2.HoughLines(edges, 1, np.pi/180, 128)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

cv2.imshow('Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()