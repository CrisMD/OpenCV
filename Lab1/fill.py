import cv2
import numpy as np

def build_solution (rows, cols, X, img):
    sol = np.ones([rows, cols])

    for i in range(rows):
        for j in range(cols):
            if (i, j) in X1 or img[i][j] == 0:
                sol[i][j] = 0

    return sol

img = cv2.imread("1.jpg", 0)
if img is None:
    print("Cannot find image")
    exit(-1)

img = cv2.resize(img, (64, 64))
cv2.imshow("Initial", cv2.resize(img, (256, 256)))

rows, cols = img.shape
retval,img = cv2.threshold(img, 180, 1, cv2.THRESH_BINARY)

with open('initial_matrix.txt', 'w') as outfile:
    for line in img:
        for item in line:
            outfile.write("%s " % item)
        outfile.write("\n")

complMat = [[0 for x in range(cols)] for y in range(rows)]
for i in range(rows):
    for j in range(cols):
        complMat[i][j] = 1 - img[i][j]

with open('complementary_matrix.txt', 'w') as outfile:
    for line in complMat:
        for item in line:
            outfile.write("%s " % item)
        outfile.write("\n")


B = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]

initial = (13, 30)
X0 = set()
X0.add(initial)
X1 = set()

while True:
    # intersect
    for point in X0:
        for dir in B:
            candidate = (point[0] + dir[0], point[1] + dir[1])
            if complMat[candidate[0]][candidate[1]] == 0:
                X1.add(candidate)

    if X1 == X0:
        break

    sol = build_solution(rows, cols, X1, img)
    cv2.imshow("res", cv2.resize(sol, (256, 256)))
    cv2.waitKey(0)

    X0 = X1
    X1 = set()

# build solution
sol = build_solution(rows, cols, X1, img)
with open('final_matrix.txt', 'w') as outfile:
    for line in sol:
        for item in line:
            outfile.write("%d " % item)
        outfile.write("\n")

cv2.imshow("Solution", cv2.resize(sol, (256, 256)))
cv2.waitKey(0)

cv2.destroyAllWindows()
