import cv2
import numpy as np

# Load the image and convert to grayscale
img = cv2.imread('./detectionTestImage/test1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 250, 450, apertureSize=3)

# Apply Hough transform to detect lines
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

# Draw the detected lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()