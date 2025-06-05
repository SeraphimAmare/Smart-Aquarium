# Import necessary libraries
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv
import numpy as np
import math


# Confidence threshold to filter out low-confidence detections
CONFIDENCE_THRESHOLD = .99  # Lowered to detect more potential fish


# Track previous positions and distances
previous_centers = []
previous_movement = float('inf')
feed = False


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Function to process predictions and print fish locations
def process_predictions(predictions, frame):
    global previous_centers, previous_movement, feed


    detections = sv.Detections.from_inference(predictions)


    if len(detections.xyxy) == 0:
        print("No fish detected in this frame.")
        previous_centers = []
        return render_boxes(predictions, frame)


    current_centers = []
    fish_count = 1
    total_movement = 0


    dets_sorted = sorted(list(zip(detections.confidence, detections.xyxy)))
    print(dets_sorted)
   
























    for (x_min, y_min, x_max, y_max), confidence in zip(detections.xyxy, detections.confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            continue


        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        current_centers.append((x_center, y_center))


        print(f"Fish {fish_count}: Center=({x_center:.2f}, {y_center:.2f}), Width={width:.2f}, Height={height:.2f}")
        fish_count += 1


    # Only compare if we have previous centers with same number of detections
    if len(current_centers) == len(previous_centers):
        for curr, prev in zip(current_centers, previous_centers):
            total_movement += calculate_distance(curr, prev)


        print(f"Total Movement: {total_movement:.2f}")
        if total_movement < previous_movement:
            feed = True
            print("🐟 Movement decreased. Feed = True")
        else:
            feed = False
            print("➖ Movement did not decrease. Feed = False")
       
        previous_movement = total_movement
    else:
        print("⚠️ Number of fish changed. Skipping movement comparison.")
        feed = False
        previous_movement = float('inf')


    previous_centers = current_centers
    return render_boxes(predictions, frame)


# Initialize the inference pipeline
pipeline = InferencePipeline.init(
    model_id="zebrafish_in_aquarium/4",
    video_reference=0,
    on_prediction=process_predictions,
    api_key="4VbrdF990dieMBW6yHSR"
)


# Start the real-time detection
pipeline.start()
pipeline.join()
