import numpy as np
import cv2
from pathlib import Path

reference_folder_number = Path('number')
reference_folder_suits = Path('suit')

def check_area(contours:np.ndarray)-> float:
    cnt_area = cv2.contourArea(contours)
    return cnt_area
    # # Filter based on area
    # if not (MIN_POSSIBLE_POKER_AREA < cnt_area < MAX_POSSIBLE_POKER_AREA):

def get_corners(contours: np.ndarray)-> np.ndarray:
    epsilon = 0.05 * cv2.arcLength(contours, True)  # Adjust epsilon for accuracy
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

def get_dict_key_with_highest_counter(tracker_dict: dict) -> list:
    result_buffer_dict = dict()
    for key, value in tracker_dict.items():
        # Split the key into pixel and data parts
        parts = key.split(', ')
        pixel = ', '.join(parts[:2])  # Extract the pixel part (first two parts)
        # Check if the pixel is already in the result or update if the value is higher
        if pixel not in result_buffer_dict or value > tracker_dict[result_buffer_dict[pixel]]:
            result_buffer_dict[pixel] = key
    final_result = [key for key in result_buffer_dict.values()]
    return final_result

def add_detection_to_dictionary(tracker_dict: dict, coord_array: list):
    for coord in coord_array:
        coordinate_y = coord[0]//10 * 10
        coordinate_x = coord[1]//10 * 10
        dict_key = ", ".join(map(str, [coordinate_y,coordinate_x,coord[2]]))
        if dict_key in tracker_dict:
            tracker_dict[dict_key] += 1 
        else:
            tracker_dict[dict_key] = 0

def process_card_suits(coord_x, coord_y, card_suits, coordinate_array):
    # print(f"Processing card number: shape={card_suits.shape}")
    reference_images = {}
    for ref_path in reference_folder_suits.glob("*.jpg"):  # 假设参考图在该文件夹
        ref_image = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        ref_image = cv2.resize(ref_image, (card_suits.shape[1], card_suits.shape[0]))
        reference_images[ref_path.stem] = ref_image  # 用文件名（不含后缀）作为键

    # 初始化最小差异值和匹配的数字
    min_diff = float('inf')
    matched_suit = None
    # 比较差异
    for name, ref_image in reference_images.items():
        xor_diff = cv2.bitwise_xor(card_suits, ref_image)
        diff_value = np.sum(xor_diff)  # 计算 XOR 差异的总和
        if diff_value < min_diff:
            min_diff = diff_value
            matched_suit = name  # 保存对应的文件名（数字）

    # 绘制数字到帧上
    if matched_suit is not None:
        # print(f"Matched digit: {matched_digit}, Difference: {min_diff}")
        coordinate_array.append([coord_x, coord_y,matched_suit])

#提取數字區域
def process_card_number(coord_x, coord_y, card_number, coordinate_array):
    reference_images = {}
    for ref_path in reference_folder_number.glob("*.jpg"):  # 假设参考图在该文件夹
        ref_image = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        ref_image = cv2.resize(ref_image, (card_number.shape[1], card_number.shape[0]))
        reference_images[ref_path.stem] = ref_image  # 用文件名（不含后缀）作为键

    # 初始化最小差异值和匹配的数字
    min_diff = float('inf')
    matched_digit = None

    # 比较差异
    for name, ref_image in reference_images.items():
        xor_diff = cv2.bitwise_xor(card_number, ref_image)
        diff_value = np.sum(xor_diff)  # 计算 XOR 差异的总和

        if diff_value < min_diff:
            min_diff = diff_value
            matched_digit = name  # 保存对应的文件名（数字）

    # 绘制数字到帧上
    if matched_digit is not None:
        coordinate_array.append([coord_x, coord_y,matched_digit])
