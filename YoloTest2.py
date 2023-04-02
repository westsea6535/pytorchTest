import cv2
import numpy as np

# Load the image
image = cv2.imread("./detectionTestImage/test1.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find the edges of the trading cards
edges = cv2.Canny(gray, 50, 200)

# Find the contours of the trading cards
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum area and maximum aspect ratio of a valid trading card
min_area = 5000
max_aspect_ratio = 3

# Loop through the contours and check if they meet the criteria of a valid trading card
trading_cards = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        continue

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    aspect_ratio = max(width, height) / min(width, height)

    if aspect_ratio > max_aspect_ratio:
        continue

    trading_card_corners = np.array([box[1], box[2], box[3], box[0]])

    # Compute the perspective transformation matrix
    target_size = (322, 502)
    dst_corners = np.array([[0, 0], [target_size[0], 0], [target_size[0], target_size[1]], [0, target_size[1]]])
    M = cv2.getPerspectiveTransform(trading_card_corners.astype(np.float32), dst_corners.astype(np.float32))

    # Apply the perspective transformation to the trading card to rectify it
    warped_card = cv2.warpPerspective(image, M, target_size)

    trading_cards.append(warped_card)

# Display the extracted trading cards
for i, trading_card in enumerate(trading_cards):
    cv2.imshow(f"Trading Card {i}", trading_card)
    cv2.waitKey(0)