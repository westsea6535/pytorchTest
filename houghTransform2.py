import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('./detectionTestImage/test1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 150)

# Set minimum length of line segment to detect
minLineLength = 100

# Set maximum allowed gap between line segment endpoints
maxLineGap = 15

# Apply probabilistic Hough transform to detect line segments
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

# Draw line segments on original img
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display img with detected line segments
cv2.imshow('Line Segments', img)
cv2.waitKey(0)
cv2.destroyAllWindows()