import cv2
import numpy as np
import time


# Define the region of interest (ROI) where the ammonia alert is located
ROI = (100, 100, 50, 50)


# Define HSV color ranges
COLOR_RANGES = {
    "yellow": ([75, 5, 160], [110, 40, 200]),         # Safe
    "light_green": ([60, 30, 150], [85, 100, 255]),  # Danger
    "light_blue": ([90, 50, 150], [105, 150, 255]),  # Danger
    "dark_blue": ([105, 100, 50], [125, 255, 150]),  # Danger
}


def detect_color(hsv_roi):
    for color, (lower, upper) in COLOR_RANGES.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv_roi, lower_np, upper_np)
        if cv2.countNonZero(mask) > 100:
            return color
    return "unknown"


# Webcam start
cap = cv2.VideoCapture(0)
last_print_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break


    x, y, w, h = ROI
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    # Calculate average HSV value
    avg_hsv = cv2.mean(hsv_roi)[:3]
    avg_hsv_int = tuple(map(int, avg_hsv))


    # Print HSV values every second
    if time.time() - last_print_time >= 1:
        print(f"Average HSV in ROI: H={avg_hsv_int[0]}, S={avg_hsv_int[1]}, V={avg_hsv_int[2]}")
        last_print_time = time.time()


    # Detect color
    detected_color = detect_color(hsv_roi)
    if detected_color == "yellow":
        label = "SAFE: Yellow"
        color = (0, 255, 255)
    elif detected_color in COLOR_RANGES:  # any known danger color
        label = f"DANGER: {detected_color.replace('_', ' ').title()}"
        color = (0, 0, 255)
    else:
        label = "DANGER: Unknown"
        color = (0, 0, 255)


    # Draw rectangle and label
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
