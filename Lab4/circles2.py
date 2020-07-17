import numpy as np
import cv2

image = cv2.imread('circles.png')
output = image.copy()
image = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image: 8-bit, single channel image. If working with a color image, convert to grayscale first.
# method: Defines the method to detect circles in images. Currently, the only implemented method is cv2.HOUGH_GRADIENT, which corresponds to the Yuen et al. paper.
# dp: This parameter is the inverse ratio of the accumulator resolution to the image resolution (see Yuen et al. for more details). Essentially, the larger the dp gets, the smaller the accumulator array gets.
# minDist: Minimum distance between the center (x, y) coordinates of detected circles. If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected. If the minDist is too large, then some circles may not be detected at all.
# param1: Gradient value used to handle edge detection in the Yuen et al. method.
# param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method. The smaller the threshold is, the more circles will be detected (including false circles). The larger the threshold is, the more circles will potentially be returned.
# minRadius: Minimum size of the radius (in pixels).
# maxRadius: Maximum size of the radius (in pixels).
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 8, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)