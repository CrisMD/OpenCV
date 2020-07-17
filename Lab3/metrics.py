from __future__ import print_function
from __future__ import division
import glob
import cv2 as cv

images = [cv.imread(image) for image in glob.glob("./images/*.jpg")]

test_image = cv.imread('./test/6.jpg')
cv.imshow("Test image", test_image)
cv.waitKey(0)
cv.destroyAllWindows()

images_hsv = []
for image in images:
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    images_hsv.append(image_hsv)

hsv_test = cv.cvtColor(test_image, cv.COLOR_BGR2HSV)

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
    hist_base = cv.calcHist([image], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    images_hist.append(hist_base)

# Parameters
# images	Source arrays. They all should have the same depth, CV_8U, CV_16U or CV_32F , and the same size. Each of them can have an arbitrary number of channels.
# channels	List of the dims channels used to compute the histogram. The first array channels are numerated from 0 to images[0].channels()-1 , the second array channels are counted from images[0].channels() to images[0].channels() + images[1].channels()-1, and so on.
# mask	Optional mask. If the matrix is not empty, it must be an 8-bit array of the same size as images[i] . The non-zero mask elements mark the array elements counted in the histogram.
# histSize	Array of histogram sizes in each dimension.
# ranges	Array of the dims arrays of the histogram bin boundaries in each dimension. When the histogram is uniform ( uniform =true), then for each dimension i it is enough to specify the lower (inclusive) boundary L0 of the 0-th histogram bin and the upper (exclusive) boundary UhistSize[i]−1 for the last histogram bin histSize[i]-1 . That is, in case of a uniform histogram each of ranges[i] is an array of 2 elements. When the histogram is not uniform ( uniform=false ), then each of ranges[i] contains histSize[i]+1 elements: L0,U0=L1,U1=L2,...,UhistSize[i]−2=LhistSize[i]−1,UhistSize[i]−1 . The array elements, that are not between L0 and UhistSize[i]−1 , are not counted in the histogram.
# accumulate	Accumulation flag. If it is set, the histogram is not cleared in the beginning when it is allocated. This feature enables you to compute a single histogram from several sets of arrays, or to update the histogram in time.
hist_test = cv.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

# 1 Correlation
# 2 Chi-Square
# 3 Intersection
# 4 Bhattacharyya distance

for compare_method in range(4):
    comparisons = []
    for hist in images_hist:
        comp = cv.compareHist(hist_test, hist, compare_method)
        comparisons.append(comp)

    if compare_method == 3:
        should_rev = False
    else:
        should_rev = True

    matches = [im for (c, im) in sorted(zip(comparisons, images), key=lambda pair: pair[0], reverse=should_rev)]

    for i in range(2):
        im = matches[i]
        cv.imshow("Match_" + str(compare_method) + "_" + str(i), cv.resize(im, (512,512)))
    cv.waitKey(0)
    cv.destroyAllWindows()
