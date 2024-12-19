import cv2
import numpy as np


# 之後拿來當作判斷輪廓面積是否為撲克牌的條件 (趴數為卡片占了幾趴的畫面)
MIN_PERCENTAGE_WIDTH_OF_POKER = 0.03
MAX_PERCENTAGE_WIDTH_OF_POKER = 0.08
MIN_PERCENTAGE_LENGTH_OF_POKER = 0.08
MAX_PERCENTAGE_LENGTH_OF_POKER = 0.15

def check_circle(countour_edges_array:np.ndarray, contour_area:np.ndarray)-> float:
    '''
    檢查偵測到輪廓是否是一個圓形，回傳一個圓形比例直
    '''
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0:
        return None  # Avoid division by zero
    # Calculate circularity
    circularity = 4 * 3.14159 * (contour_area / (perimeter * perimeter))
    return circularity

# Load the video
video = cv2.VideoCapture('vid1.mkv')  # Replace 'video.mp4' with your video path or use 0 for webcam
ret, frame = video.read()
video_width, video_length, depth = frame.shape

MIN_POSSIBLE_POKER_AREA = (video_width*MIN_PERCENTAGE_WIDTH_OF_POKER) * (video_length*MIN_PERCENTAGE_LENGTH_OF_POKER)
MAX_POSSIBLE_POKER_AREA = (video_width*MAX_PERCENTAGE_WIDTH_OF_POKER) * (video_length*MAX_PERCENTAGE_LENGTH_OF_POKER)

# Check if the video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    # If the frame was not grabbed, end of video is reached
    if not ret:
        break
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter card-like contours based on area
    card_bounding_rectangles = np.zeros((52,4))
    frame_with_labels = frame.copy()

    # # Create a blank mask
    # mask = np.zeros_like(thresholded)
    # # Draw the contours onto the mask
    # cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Fill the contours with white
    # # Define a kernel for erosion
    # kernel = np.ones((9, 9), np.uint8)  # Adjust the kernel size based on your requirements
    # # Erode the mask
    # dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    # eroded_mask = cv2.erode(dilated_mask, kernel=kernel, iterations=1)
    # # Visualize the eroded contour edges
    # output = cv2.bitwise_and(frame_with_labels, frame_with_labels, mask=eroded_mask)
    # cv2.imshow('Contours with Labels', output)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    rectangle_index_counter = 0
    for id, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        cnt_area = cv2.contourArea(cnt)
        # Filter based on area
        if not (MIN_POSSIBLE_POKER_AREA < cnt_area < MAX_POSSIBLE_POKER_AREA):
            continue
        
        circularity = check_circle(cnt, cnt_area)
        # Define a threshold to classify as a circle
        if 0.75 < circularity < 1.25:  # Circularity close to 1 indicates a circle
            continue

        # Approximate the contour
        epsilon = 0.06 * cv2.arcLength(cnt, True)  # Adjust the epsilon value for precision
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for point in approx:
            approx_x, approx_y = point[0]
            cv2.circle(frame_with_labels, (approx_x, approx_y), 5, (0, 0, 255), -1)  # Draw corners

        # # Define a kernel for erosion
        # kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size for the desired effect
        # # Apply erosion
        # eroded = cv2.erode(contours, kernel, iterations=1)
        
        card_bounding_rectangles[rectangle_index_counter] = np.array([x,y,w,h])
        rectangle_index_counter += 1

        # Draw the contour
        # cv2.drawContours(frame_with_labels, [cnt], -1, (0, 255, 0), 3)
        # cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Get the bounding rectangle to position the label
        label_position = (x, y - 10)  # Position the label slightly above the contour

        # Add a label for the contour (e.g., "Card 1", "Card 2")
        cv2.putText(frame_with_labels, f'Card {id}', label_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for bounding_rects in card_bounding_rectangles:
        if not np.any(bounding_rects):
            break
        x,y,w,h = bounding_rects.astype(np.int16)
        # Apply edge detection (Canny)
        _, card_thresholded = cv2.threshold(gray[y:y+h,x:x+w], 220, 255, cv2.THRESH_BINARY)
        # Find contours
        card_contours, _ = cv2.findContours(card_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # since there might be multiple contour, we get the biggest contour
        largest_contour = max(card_contours, key=lambda arr: arr.size)
        # offsetting to the current main frame
        largest_contour[:,:,0] += y
        largest_contour[:,:,1] += x
        # cv2.drawContours(frame_with_labels,[largest_contour],1, (0, 255, 0), 3)
    
    # Display the frame with contours and labels
    cv2.imshow('Contours with Labels', frame_with_labels)
    frame_with_labels
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()