import cv2
import numpy as np
import funcs
from pathlib import Path
video_file = 'self_video_2.mp4'
video = cv2.VideoCapture(video_file)  
ret, frame = video.read()

video_width, video_length, depth = frame.shape
MIN_PERCENTAGE_WIDTH_OF_POKER = 0.05
MAX_PERCENTAGE_WIDTH_OF_POKER = 0.2
MIN_PERCENTAGE_LENGTH_OF_POKER = 0.1
MAX_PERCENTAGE_LENGTH_OF_POKER = 0.3
MIN_POSSIBLE_POKER_AREA = (video_width*MIN_PERCENTAGE_WIDTH_OF_POKER) * (video_length*MIN_PERCENTAGE_LENGTH_OF_POKER)
MAX_POSSIBLE_POKER_AREA = (video_width*MAX_PERCENTAGE_WIDTH_OF_POKER) * (video_length*MAX_PERCENTAGE_LENGTH_OF_POKER)
print(MIN_POSSIBLE_POKER_AREA, MAX_POSSIBLE_POKER_AREA)
output_folder = Path('output_images')
# Ensure the folder exists
output_folder.mkdir(parents=True, exist_ok=True)

while True:
    ret, frame = video.read()
    # 灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255,  # Max value
        cv2.ADAPTIVE_THRESH_MEAN_C,  # Gaussian method for adaptive threshold
        cv2.THRESH_BINARY_INV,  # Threshold type
        23, 9  # Block size and constant
    )
    
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for id, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        contour_area = funcs.check_area(cnt)
        if not (MIN_POSSIBLE_POKER_AREA < contour_area < MAX_POSSIBLE_POKER_AREA):
            continue
        corners = funcs.get_corners(cnt)
        if corners is None:
            continue
        card_straight = funcs.warp_image(adaptive_thresh, 200, 300, corners)

        card_border = card_straight[:int(300*0.25),:int(200*0.2)]

        # 花紋比對寫在這
        

        # output_path = output_folder / f'{video_file}_{id}_border.jpg'
        # cv2.imwrite(str(output_path),card_border)


    # Display the frame with contours
    cv2.imshow('Contours on Video', frame)
    # Display the frame with contours and labels
    cv2.imshow('Contours with Labels', adaptive_thresh)
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

