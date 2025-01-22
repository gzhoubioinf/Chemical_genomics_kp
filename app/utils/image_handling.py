import cv2
import numpy as np

def crop_img(image_path):
    """Reads an image from disk in BGR format."""
    return cv2.imread(image_path)

def extract_colony(img, row, col, w, h):
    """
    Extract the grid cell at (row, col), find the largest contour
    that represents the colony outline, and re-center so the entire colony
    is in the middle of the extracted snippet.
    """
    # Coordinates of the raw sub-cell
    x = col * w
    y = row * h
    cell = img[y:y + h, x:x + w]

    # Ensure the extracted cell matches the expected dimensions
    if cell.shape[:2] != (h, w):
        cell = cv2.resize(cell, (w, h))

    if cell.size == 0:
        return None

    # Turquoise color range(s)
    lower_bound = np.array([130 - 40, 200 - 40, 70 - 40])  # Approx (B, G, R)
    upper_bound = np.array([230 + 40, 255, 180 + 40])

    # Clamp to [0..255]
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)

    # Create mask for contours
    mask = cv2.inRange(cell, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No contours detected; return the original unmodified cell, resized if needed
        return cell

    # Pick the largest contour as the main colony
    colony_contour = max(contours, key=cv2.contourArea)
    xC, yC, wC, hC = cv2.boundingRect(colony_contour)

    # Center of bounding box in the sub-cell
    cX = xC + wC // 2
    cY = yC + hC // 2

    # Center of the snippet
    sub_center_x = w // 2
    sub_center_y = h // 2
    shiftX = cX - sub_center_x
    shiftY = cY - sub_center_y

    # Adjust the top-left corner based on that shift
    startX = max(0, min(x + shiftX, img.shape[1] - w))
    startY = max(0, min(y + shiftY, img.shape[0] - h))
    endX = startX + w
    endY = startY + h

    centered_cell = img[startY:endY, startX:endX]

    # Ensure the final output matches the expected dimensions
    if centered_cell.shape[:2] != (h, w):
        centered_cell = cv2.resize(centered_cell, (w, h))

    return centered_cell