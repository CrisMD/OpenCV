from __future__ import print_function
from __future__ import division
from sklearn.cluster import MiniBatchKMeans
import glob
import cv2
import numpy as np

images = [cv2.imread(image) for image in glob.glob("./images/*.jpg")]

test_image = cv2.imread('./test/4.jpg')
cv2.imshow("Test image", test_image)

def color_reduction(image):
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=16)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return quant


images_reduced = []
for image in images:
    images_reduced.append(color_reduction(image))

images_hsv = []
for image in images_reduced:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images_hsv.append(image_hsv)

hsv_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]

images_hist = []
for image in images_hsv:
    hist_base = cv2.calcHist([image], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    images_hist.append(hist_base)

hist_test = cv2.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

for compare_method in range(4):
    comparisons = []
    for hist in images_hist:
        comp = cv2.compareHist(hist_test, hist, compare_method)
        comparisons.append(comp)

    if compare_method == 1 or compare_method == 3:
        should_rev = False
    else:
        should_rev = True

    matches = [im for (c, im) in sorted(zip(comparisons, images), key=lambda pair: pair[0], reverse=should_rev)]

    for i in range(2):
        im = matches[i]
        cv2.imshow("Match_" + str(compare_method) + "_" + str(i), im)

cv2.waitKey(0)
cv2.destroyAllWindows()

