import cv2
import numpy as np

def crop_img(image_path):
    """ loads an image from disk using OpenCV and returns it as a NumPy array in BGR order."""
    return cv2.imread(image_path)

def extract_colony(img, row, col, num_rows=32, num_cols=48):
    """treats the plate image as a num_rows x num_cols grid, computes the pixel boundaries of the specified 
    cell, and crops that region; it then detects the turquoise IRIS outline, finds the largest 
    matching contour (the colony), and shifts the crop so the colony is centred while keeping the 
    snippet within the original image bounds, finally returning the centred cell—or None if the extraction 
    cannot proceed."""
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Calculate cell dimensions based on full grid size
    cell_height = img_height / num_rows  
    cell_width = img_width / num_cols 
    
    # Calculate position using floating-point precision
    x = col * cell_width
    y = row * cell_height
    
    # Convert to integers with boundary checks
    x_start = int(round(x))
    y_start = int(round(y))
    x_end = int(round(x + cell_width))
    y_end = int(round(y + cell_height))

    # Ensure we stay within image bounds
    x_start = max(0, min(x_start, img_width - 2))
    y_start = max(0, min(y_start, img_height - 2))
    x_end = max(x_start + 1, min(x_end, img_width))
    y_end = max(y_start + 1, min(y_end, img_height))

    # Extract base cell
    cell = img[y_start:y_end, x_start:x_end]

    # Rest of the original processing logic
    if cell.size == 0:
        return None

    # Turquoise color range detection - colonies detected by iris are delineated with this color 
    lower_bound = np.array([130 - 40, 200 - 40, 70 - 40])
    upper_bound = np.array([230 + 40, 255, 180 + 40])
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)

    mask = cv2.inRange(cell, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return cell

    # Contour processing (keep original code)
    colony_contour = max(contours, key=cv2.contourArea)
    xC, yC, wC, hC = cv2.boundingRect(colony_contour)
    
    # Recenter
    cX = xC + wC // 2
    cY = yC + hC // 2
    sub_center_x = cell.shape[1] // 2  
    sub_center_y = cell.shape[0] // 2  
    
    shiftX = cX - sub_center_x
    shiftY = cY - sub_center_y

    # Calculate new coordinates in original image
    new_x = x_start + shiftX
    new_y = y_start + shiftY
    
    # Ensure we don't go out of bounds
    new_x = max(0, min(new_x, img_width - cell.shape[1]))
    new_y = max(0, min(new_y, img_height - cell.shape[0]))
    
    centered_cell = img[new_y:new_y+cell.shape[0], new_x:new_x+cell.shape[1]]
    
    return centered_cell