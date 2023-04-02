import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('./detectionTestImage/test2.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to remove noise while preserving edges
filtered_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(filtered_image, 100, 200)

# Apply binary thresholding to convert the edges to a binary image
ret, binary_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

# Define a 5x5 kernel for dilation and erosion
kernel = np.ones((3,3), np.uint8)

# Dilate the binary image to fill in gaps in edges
dilated_image = cv2.dilate(binary_image, kernel, iterations=3)

# Erode the dilated image to restore edges to their original thickness
eroded_image = cv2.erode(dilated_image, kernel, iterations=3)

onImage = True

if onImage:
  # Wait for a key press and then close all windows
  plt.subplot(2, 4, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 2), plt.imshow(gray_image, cmap='gray'), plt.title('Grayscale')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 3), plt.imshow(filtered_image, cmap='gray'), plt.title('bilateralFilter')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 4), plt.imshow(edges, cmap='gray'), plt.title('Canny')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 5), plt.imshow(binary_image, cmap='gray'), plt.title('threshold')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 6), plt.imshow(dilated_image, cmap='gray'), plt.title('dilate')
  plt.xticks([]), plt.yticks([])
  plt.subplot(2, 4, 7), plt.imshow(eroded_image, cmap='gray'), plt.title('erode')
  plt.xticks([]), plt.yticks([])
  plt.show()
else:
  findContour = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  grab = imutils.grab_contours(findContour)

  sort = sorted(grab, key = cv2.contourArea, reverse = True)[:10]

  # Initialize variable to store the contour of the screen
  screenCnt = None

  # Draw the contours on the original image
  cv2.drawContours(image, sort, -1, (0, 255, 0), 3)

  # Display the image with contours
  cv2.imshow("Image with Contours", image)
  cv2.waitKey(0)