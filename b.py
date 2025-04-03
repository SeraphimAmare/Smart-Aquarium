from inference import get_model
import supervision as sv
import cv2

model = get_model(model_id="objectdetectionfish/1", api_key="4VbrdF990dieMBW6yHSR")

image_file = "images.jpg"
image = cv2.imread(image_file)

results = model.infer(image)[0]

detections = sv.Detections.from_inference(results)

for i, (x_min, y_min, x_max, y_max) in enumerate(detections.xyxy):
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    print(f"Fish {i + 1}: Center=({x_center}, {y_center}), Width={width}, Height={height}")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
