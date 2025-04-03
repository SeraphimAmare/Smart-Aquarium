from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv

CONFIDENCE_THRESHOLD = 0.89  

def process_predictions(predictions, frame):
    """
    Processes model predictions in real-time, extracting fish locations
    and filtering low-confidence detections.
    """
    detections = sv.Detections.from_inference(predictions)

    if len(detections.xyxy) == 0:
        print("No fish detected in this frame.")
        return render_boxes(predictions, frame)

    fish_count = 1
    for (x_min, y_min, x_max, y_max), confidence in zip(detections.xyxy, detections.confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            continue  
      
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        print(f"Fish {fish_count}: Center=({x_center:.2f}, {y_center:.2f}), Width={width:.2f}, Height={height:.2f}")
        fish_count += 1

    return render_boxes(predictions, frame)  

pipeline = InferencePipeline.init(
    model_id="zebrafish_in_aquarium/4",  
    video_reference=0,  
    on_prediction=process_predictions, 
)

# Start the real-time detection
pipeline.start()
pipeline.join()
