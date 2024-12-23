import cv2
image = cv2.imread('img1.png')
# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # (15, 15) is the kernel size

gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Gray', thresholded)

# Filter card-like contours based on area and aspect ratio
card_contours = []
print(gray.size)

for id, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    if 3000 < cv2.contourArea(cnt) < 40000:
        card_contours.append(cnt)

# Draw contours and label each one
image_with_labels = image.copy()
for idx, cnt in enumerate(card_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h

    print(idx, aspect_ratio, cv2.contourArea(cnt))

    # Draw the contour
    cv2.drawContours(image_with_labels, [cnt], -1, (0, 255, 0), 3)

    # Get the bounding rectangle to position the label
    x, y, w, h = cv2.boundingRect(cnt)
    label_position = (x, y - 10)  # Position the label slightly above the contour

    # Add a label for the contour (e.g., "Card 1", "Card 2")
    cv2.putText(image_with_labels, f'Card {idx + 1}', label_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the contours with labels
cv2.imshow('Contours with Labels', image_with_labels)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
