import cv2

logoimg = cv2.imread("image.png")
if logoimg is None:
    print("Cannot find logo image")
    exit(-1)

sceneimg = cv2.imread("lenna.png")
if sceneimg is None:
    print("Cannot find scene image")
    exit(-1)

# logoimg = cv2.resize(logoimg, (512, 512))
logoimg = cv2.resize(logoimg,(0,0), fx=1,fy=1)
logo_mask = cv2.cvtColor(logoimg,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Result", logo_mask)
# cv2.waitKey(0)

retval,logo_mask = cv2.threshold(logo_mask,180,255,cv2.THRESH_BINARY_INV)

roi = sceneimg[0:logo_mask.shape[0], sceneimg.shape[1] - logo_mask.shape[1]:sceneimg.shape[1]]
cv2.add(logoimg, roi, roi, logo_mask)
cv2.imshow("Result", sceneimg)
cv2.waitKey(0)

cv2.destroyAllWindows()