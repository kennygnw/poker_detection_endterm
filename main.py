import cv2
import numpy as np
import funcs
from pathlib import Path
import time



video_file = 'video9.mp4'
video = cv2.VideoCapture(video_file)
ret, frame = video.read()
reference_folder_number = Path('number')

video_width, video_length, depth = frame.shape
MIN_PERCENTAGE_WIDTH_OF_POKER = 0.05
MAX_PERCENTAGE_WIDTH_OF_POKER = 0.2
MIN_PERCENTAGE_LENGTH_OF_POKER = 0.1
MAX_PERCENTAGE_LENGTH_OF_POKER = 0.3
MIN_POSSIBLE_POKER_AREA = (video_width * MIN_PERCENTAGE_WIDTH_OF_POKER) * (
            video_length * MIN_PERCENTAGE_LENGTH_OF_POKER)
MAX_POSSIBLE_POKER_AREA = (video_width * MAX_PERCENTAGE_WIDTH_OF_POKER) * (
            video_length * MAX_PERCENTAGE_LENGTH_OF_POKER)

WARP_HEIGHT_PIXEL = 300
WARP_LENGTH_PIXEL = 200


reference_folder_number = Path('number')
reference_folder_suits = Path('suit')
# reference_number = cv2.imread('reference_number.jpg', cv2.IMREAD_GRAYSCALE)

pattern_coord_array = list()
number_coord_array = list()
pattern_coord_tracker_dict = dict()
number_coord_tracker_dict = dict()
number_final_result = list()
pattern_final_result = list()
update_label_counter = np.uint8(0)
update_label_frame_limit = np.uint8(10)
output_dataset_buffer_path = Path('dataset_buffer')
imwrite_counter = 0
while True:
    ret, frame = video.read()
    # 灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255,  # Max value
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,  # Threshold type
        11, 3  # Block size and constant
    )
    # filename = f"{imwrite_counter}.jpg"
    # cv2.imwrite(str(output_dataset_buffer_path / filename), frame)
    # imwrite_counter += 1
    # filename = f"{imwrite_counter}.jpg"
    # cv2.imwrite(str(output_dataset_buffer_path / filename), adaptive_thresh)
    # imwrite_counter += 1

    # 著輪廓
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 偵測到輪廓迴圈
    for id, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # 判斷面積
        contour_area = funcs.check_area(cnt)
        if not (MIN_POSSIBLE_POKER_AREA < contour_area < MAX_POSSIBLE_POKER_AREA):
            continue
        # 取邊角
        corners = funcs.get_corners(cnt)
        if corners is None:
            continue
        for c in corners:
            cv2.circle(frame, (int(c[0]), int(c[1])), 3, (255, 0, 0), 3, 1)
        # 透視變換
        card_straight = funcs.warp_image(adaptive_thresh, WARP_LENGTH_PIXEL, WARP_HEIGHT_PIXEL, corners)
        # 取左上角
        card_border = card_straight[:int(WARP_HEIGHT_PIXEL * 0.3), :int(WARP_LENGTH_PIXEL * 0.25)]
        height, width = card_border.shape[:2]
        # 花色區域
        pattern_roi_start_x = int(width * 0.2)
        pattern_roi_start_y = int(height * 0.5)
        pattern_roi_end_x = int(width * 0.8)
        pattern_roi_end_y = int(height * 1)
        pattern_roi = card_border[pattern_roi_start_y:pattern_roi_end_y, pattern_roi_start_x:pattern_roi_end_x]
        pattern_roi_contour, _ = cv2.findContours(pattern_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_border_contour, _ = cv2.findContours(card_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('cb', card_border)
        # cv2.imshow('pr', pattern_roi)
        # 提取花色區域
        for pattern_contour in pattern_roi_contour:
            x2, y2, w2, h2 = cv2.boundingRect(pattern_contour)
            pattern_contour_area = w2 * h2
            if  400 <= pattern_contour_area < 800 :
                pattern_contour_area_confirmed = pattern_roi[y2:y2 + h2, x2:x2 + w2]
                funcs.process_card_suits(x, y, pattern_contour_area_confirmed, pattern_coord_array)

        # 用字典追蹤每個frame偵測到的結果
        if len(pattern_coord_array) != 0:
            funcs.add_detection_to_dictionary(pattern_coord_tracker_dict, pattern_coord_array)

        for number_contour in card_border_contour:
            x2, y2, w2, h2 = cv2.boundingRect(number_contour)
            number_contour_area = h2 * w2
            if  700 <=number_contour_area < 1000 :
                number_contour_area_confirmed= card_border[y2:y2 + h2, x2:x2 + w2]
                # filename = f"{imwrite_counter}.jpg"
                # cv2.imwrite(str(output_dataset_buffer_path/filename),number_contour_area_confirmed)
                # imwrite_counter += 1
                funcs.process_card_number(x, y, number_contour_area_confirmed, number_coord_array)

        if len(number_coord_array) != 0:
            funcs.add_detection_to_dictionary(number_coord_tracker_dict, number_coord_array)
    
        #  更新追蹤計數器
        update_label_counter += 1
        # 到了N個frame後更新畫面上有哪些被偵測
        if update_label_counter >= update_label_frame_limit:
            number_final_result = funcs.get_dict_key_with_highest_counter(number_coord_tracker_dict)
            pattern_final_result = funcs.get_dict_key_with_highest_counter(pattern_coord_tracker_dict)
            update_label_counter = 0
            number_coord_tracker_dict.clear()
            pattern_coord_tracker_dict.clear()
    # 上字
    for filtered_number in number_final_result:
        filtered_number_data = filtered_number.split(', ')
        cv2.putText(frame, filtered_number_data[2], (int(filtered_number_data[0]), int(filtered_number_data[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for filtered_pattern in pattern_final_result:
        filtered_pattern_data = filtered_pattern.split(', ')
        cv2.putText(frame, filtered_pattern_data[2], (int(filtered_pattern_data[0])+30, int(filtered_pattern_data[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Display the frame with contours
    cv2.imshow('Contours on Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pattern_coord_array.clear()
    number_coord_array.clear()
    
# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()