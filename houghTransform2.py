import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread('./detectionTestImage/test3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Set minimum length of line segment to detect
minLineLength = 50

# Set maximum allowed gap between line segment endpoints
maxLineGap = 15

# Apply probabilistic Hough transform to detect line segments
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=minLineLength, maxLineGap=maxLineGap)


# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     # Calculate angle of the line segment
#     angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#     # Check if the angle is within a certain range
#     if abs(angle) < 30 or abs(angle - 90) < 30:
#         filtered_lines.append(line)

# Draw line segments on original img
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Display img with detected line segments
cv2.imshow('Line Segments', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
