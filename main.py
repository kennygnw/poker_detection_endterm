import cv2
import numpy as np
import funcs
from pathlib import Path
import time

def process_card_suits(card_suits, coordinate_array):
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

        # print(f"Template: {name}, Difference: {diff_value}")

        if diff_value < min_diff:
            min_diff = diff_value
            matched_suit = name  # 保存对应的文件名（数字）

    # 绘制数字到帧上
    if matched_suit is not None:
        # print(f"Matched digit: {matched_digit}, Difference: {min_diff}")
        coordinate_array.append([x,y,matched_suit])
        # # 假设需要在 card_number 的左上角显示数字
        # cv2.putText(
        #     frame,
        #     matched_digit,
        #     (x+30, y - 5),  #
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,  # 字体大小
        #     (255, 0, 0),  # 颜色
        #     2  # 粗细
        # )

#提取數字區域
def process_card_number(card_number, coordinate_array):
    # print(f"Processing card number: shape={card_number.shape}")
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

        # print(f"Template: {name}, Difference: {diff_value}")

        if diff_value < min_diff:
            min_diff = diff_value
            matched_digit = name  # 保存对应的文件名（数字）

    # 绘制数字到帧上
    if matched_digit is not None:
        coordinate_array.append([x,y,matched_digit])
        # print(f"Matched digit: {matched_digit}, Difference: {min_diff}")
        # 假设需要在 card_number 的左上角显示数字
        # cv2.putText(
        #     frame,
        #     matched_digit,
        #     (x, y - 5),  #
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,  # 字体大小
        #     (255, 0, 0),  # 颜色
        #     2  # 粗细
        # )

video_file = 'video5.mp4'
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


output_folder1 = Path('output_number')
output_folder2= Path('output_suits')
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
# Ensure the folder exists
output_folder1.mkdir(parents=True, exist_ok=True)
output_folder2.mkdir(parents=True, exist_ok=True)

while True:
    ret, frame = video.read()
    # 灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255,  # Max value
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,  # Threshold type
        11, 9  # Block size and constant
    )
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # frame = cv2.drawContours(frame,contours, -1, (0,0,255),2)
    for id, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        contour_area = funcs.check_area(cnt)
        if not (MIN_POSSIBLE_POKER_AREA < contour_area < MAX_POSSIBLE_POKER_AREA):
            continue
        # cv2.putText(frame,f"{int(contour_area)}",(x,y-10),1,1,(0,255,0),2)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        corners = funcs.get_corners(cnt)
        if corners is None:
            continue
        for c in corners:
            # print(c)
            cv2.circle(frame, (int(c[0]), int(c[1])), 3, (255, 0, 0), 3, 1)
        card_straight = funcs.warp_image(adaptive_thresh, WARP_LENGTH_PIXEL, WARP_HEIGHT_PIXEL, corners)
        # cv2.imshow('warpperspective',card_straight)
        # cv2.waitKey(1)
        card_border = card_straight[:int(WARP_HEIGHT_PIXEL * 0.3), :int(WARP_LENGTH_PIXEL * 0.25)]
        height, width = card_border.shape[:2]
        start_x = int(width * 0.2)
        start_y = int(height * 0.5)
        end_x = int(width * 0.8)
        end_y = int(height * 1)
        sub_region = card_border[start_y:end_y, start_x:end_x]
        # cv2.imshow('Sub Region', sub_region)
        # card_border_bottom = card_straight[int(WARP_HEIGHT_PIXEL*(1-0.3)):,int(WARP_LENGTH_PIXEL*(1-0.25)):]
        # cv2.imshow('cardborder', card_border)

        # cv2.moveWindow('cardborder', 40,500)
        #cv2.imshow('cardborderbottom', card_border_bottom)

        border_contour_array, _ = cv2.findContours(sub_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_border_contour1, _ = cv2.findContours(card_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 提取花色區域
        for pattern_contour in border_contour_array:
            x2, y2, w2, h2 = cv2.boundingRect(pattern_contour)
            area = w2 * h2
            if  400 <=area < 800 :
                cv2.rectangle(sub_region, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), 2)
                card_suits = sub_region[y2:y2 + h2, x2:x2 + w2]
                # cv2.imshow('cardborder', sub_region[y2:y2 + h2, x2:x2 + w2])
                # output_path = output_folder2 / f'{video_file}_{id}_border.jpg'
                # cv2.imwrite(str(output_path), sub_region[y2:y2 + h2, x2:x2 + w2])
                process_card_suits(card_suits, pattern_coord_array)
        
        if len(pattern_coord_array) != 0:
            for pattern_coord in pattern_coord_array:
                coordinate_y = pattern_coord[0]//10 * 10
                coordinate_x = pattern_coord[1]//10 * 10
                dict_key = ", ".join(map(str, [coordinate_y,coordinate_x,pattern_coord[2]]))
                if dict_key in pattern_coord_tracker_dict:
                    pattern_coord_tracker_dict[dict_key] += 1 
                else:
                    pattern_coord_tracker_dict[dict_key] = 0

                # cv2.putText(frame, pattern_coord[2], (pattern_coord[0]+30, pattern_coord[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        for a in card_border_contour1:
            x2, y2, w2, h2 = cv2.boundingRect(a)

            area = h2 * w2
            # print(area)
            if  700 <=area < 1000 :
                cv2.rectangle(card_border, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), 2)
                card_number= card_border[y2:y2 + h2, x2:x2 + w2]
                # cv2.imshow('cardborder2', card_border[y2:y2 + h2, x2:x2 + w2])
                # output_path = output_folder1 / f'{video_file}_{id}_border.jpg'
                # cv2.imwrite(str(output_path), card_border[y2:y2 + h2, x2:x2 + w2])
                process_card_number(card_number, number_coord_array)
                cv2.waitKey(1)

        if len(number_coord_array) != 0:
            for number_coord in number_coord_array:
                coordinate_y = number_coord[0]//10 * 10
                coordinate_x = number_coord[1]//10 * 10
                dict_key = ", ".join(map(str, [coordinate_y,coordinate_x,number_coord[2]]))
                if dict_key in number_coord_tracker_dict:
                    number_coord_tracker_dict[dict_key] += 1 
                else:
                    number_coord_tracker_dict[dict_key] = 0
        update_label_counter += 1
        if update_label_counter >= 15:
            result_buffer = {}
            for key, value in number_coord_tracker_dict.items():
                # Split the key into pixel and data parts
                parts = key.split(', ')
                pixel = ', '.join(parts[:2])  # Extract the pixel part (first two parts)
                # Check if the pixel is already in the result or update if the value is higher
                if pixel not in result_buffer or value > number_coord_tracker_dict[result_buffer[pixel]]:
                    result_buffer[pixel] = key
            # Create the final dictionary with only the reserved keys
            number_final_result = [key for key in result_buffer.values()]
            result_buffer.clear()
            for key, value in pattern_coord_tracker_dict.items():
                # Split the key into pixel and data parts
                parts = key.split(', ')
                pixel = ', '.join(parts[:2])  # Extract the pixel part (first two parts)
                # Check if the pixel is already in the result or update if the value is higher
                if pixel not in result_buffer or value > pattern_coord_tracker_dict[result_buffer[pixel]]:
                    result_buffer[pixel] = key
            # Create the final dictionary with only the reserved keys
            pattern_final_result = [key for key in result_buffer.values()]

            # Get keys where values are greater than 5
            # key_list_with_values_gt_5 = [key for key, value in number_coord_tracker_dict.items() if value > 5]
            # for key in key_list_with_values_gt_5:
                # filtered_number_coord = key.split(', ')
                # cv2.putText(frame, filtered_number_coord[2], (int(filtered_number_coord[0]), int(filtered_number_coord[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            update_label_counter = 0
            number_coord_tracker_dict.clear()
            pattern_coord_tracker_dict.clear()
    for filtered_number in number_final_result:
        filtered_number_data = filtered_number.split(', ')
        cv2.putText(frame, filtered_number_data[2], (int(filtered_number_data[0]), int(filtered_number_data[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for filtered_pattern in pattern_final_result:
        filtered_pattern_data = filtered_pattern.split(', ')
        cv2.putText(frame, filtered_pattern_data[2], (int(filtered_pattern_data[0])+30, int(filtered_pattern_data[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Display the frame with contours
    cv2.imshow('Contours on Video', frame)

    # Display the frame with contours and labels
    # cv2.imshow('Contours with Labels', adaptive_thresh)
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pattern_coord_array.clear()
    number_coord_array.clear()
# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()