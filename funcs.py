import numpy as np
import cv2



def check_area(contours:np.ndarray)-> float:
    cnt_area = cv2.contourArea(contours)
    return cnt_area
    # # Filter based on area
    # if not (MIN_POSSIBLE_POKER_AREA < cnt_area < MAX_POSSIBLE_POKER_AREA):

def get_corners(contours: np.ndarray)-> np.ndarray:
    epsilon = 0.02 * cv2.arcLength(contours, True)  # Adjust epsilon for accuracy
    approx = cv2.approxPolyDP(contours, epsilon, True)
    if len(approx) == 4:
        corners = approx.reshape(4, 2)  # Extract the (x, y) points of the corners
        corners = np.array(corners, dtype=np.float32)
        return corners
    else:
        return None   
    
def warp_image(full_img:cv2.typing.MatLike, target_width:int, target_height:int, given_corners:np.ndarray)->cv2.UMat:
    '''
    given_corners should be [4,2] float32 array
    '''
    is_corner_landscape = check_if_landscape(given_corners)
    if is_corner_landscape:
        dst_points = np.array([
            [0, 0],                 # Top-left
            [target_height - 1, 0],        # Top-right
            [target_height - 1, target_width - 1],  # Bottom-right
            [0, target_width - 1]          # Bottom-left
        ], dtype=np.float32)
    else:
        dst_points = np.array([
        [0, 0],                # Top-left corner
        [target_width - 1, 0],        # Top-right corner
        [target_width - 1, target_height - 1],  # Bottom-right corner
        [0, target_height - 1]        # Bottom-left corner
        ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(given_corners, dst_points)
    # Apply the perspective warp
    if is_corner_landscape:
        warped_image = cv2.warpPerspective(full_img, M, (int(target_height), int(target_width)))
         # Rotate the original image by 90 degrees clockwise
        warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        warped_image = cv2.warpPerspective(full_img, M, (target_width, target_height))
    # Flip the image horizontally
    flipped_image = cv2.flip(warped_image, 1)
    return flipped_image

def check_if_landscape(given_corners:np.ndarray)-> bool:
    # Compute width and height
    corner_width = np.linalg.norm(given_corners[1] - given_corners[0])  # Distance between top-left and top-right
    corner_height = np.linalg.norm(given_corners[3] - given_corners[0])  # Distance between top-left and bottom-left
    if corner_width > corner_height:
        return True
    else:
        return False
