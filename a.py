from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="4VbrdF990dieMBW6yHSR"
)

result = CLIENT.infer("images.jpg", model_id="objectdetectionfish/1")

print(result)