import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread('scarlett1.jpg')
image2 = cv2.imread('scarlett2.jpg')

# Convert the images to RGB
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Display traning image and testing image
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Training Image")
plots[0].imshow(training_image)

plots[1].set_title("Testing Image")
plots[1].imshow(test_image)

# Modify test image by adding Scale Invariance and Rotational Invariance
test_image = cv2.pyrDown(test_image)
test_image = cv2.pyrDown(test_image)
num_rows, num_cols = test_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

# Convert the images to gray scale
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

# Apply SURF algorithm with Threshold for hessian keypoint detector = 800
surf = cv2.xfeatures2d.SURF_create(800)

train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None)

# Draw keypoints for training image
keypoints_without_size = np.copy(training_image)
keypoints_with_size = np.copy(training_image)
cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))
cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw keypoints for testing image
keypoints_without_size_test = np.copy(test_image)
keypoints_with_size_test = np.copy(test_image)

cv2.drawKeypoints(test_image, test_keypoints, keypoints_without_size_test, color = (0, 255, 0))
cv2.drawKeypoints(test_image, test_keypoints, keypoints_with_size_test, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with and without keypoints size
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("Train keypoints")
plots[0].imshow(keypoints_with_size, cmap='gray')

plots[1].set_title("Test keypoints")
plots[1].imshow(keypoints_with_size_test, cmap='gray')

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

# Create a Brute Force Matcher object.
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

# Perform the matching between the SURF descriptors of the training image and the test image
matches = bf.match(train_descriptor, test_descriptor)

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)

# Display the best matching points
fx, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title('Best Matching Points')
plots[0].imshow(result)
plots[1].set_title('Original query image')
plots[1].imshow(test_image)
plt.show()

# Print total number of matching points between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
