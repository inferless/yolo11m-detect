import os
import requests
import torch
import inferless
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
import uuid
import cv2
import base64

def download_file(url: str, save_path: str, overwrite: bool = False) -> bool:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not overwrite and os.path.exists(save_path):
        print(f"File already exists at {save_path}, skipping download.")
        return True

    print(f"Attempting to download: {url} to {save_path}")
    response = requests.get(url, stream=True, timeout=30) # Added timeout
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # Increased block size for potentially faster downloads

    with open(save_path, 'wb') as f:
        downloaded_size = 0
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded_size += len(data)
            if total_size > 0:
                done = int(50 * downloaded_size / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='')
    print("\nDownload complete!")
    return True

@inferless.request
class RequestObjects(BaseModel):
    image_url: str = Field(description="URL of the image to perform detection on.")
    confidence_threshold: Optional[float] = Field(default=0.25)
    # Add any other YOLO specific inference parameters you might need
    # For example:
    # iou_threshold: Optional[float] = Field(default=0.45)

@inferless.response
class ResponseObject(BaseModel):
    boxes: List[float] = []
    confidences: List[float] = []
    class_ids: List[float]= []
    class_names: List[str] = []

class InferlessPythonModel:
    def initialize(self, context=None):
        self.model_weights_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
        models_dir = os.path.join(os.getcwd(), "models_yolo")
        self.model_save_path = os.path.join(models_dir, "yolo11m.pt")
        self.temp_image_dir = os.path.join(os.getcwd(), "temp_images")

        print("Initializing YOLO model...")
        if download_file(self.model_weights_url, self.model_save_path, overwrite=False):
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading YOLO model from {self.model_save_path} onto {self.device}")
                self.model = YOLO(self.model_save_path)
                self.model.to(self.device) # Ensure model is on the correct device
                print("YOLO model loaded successfully.")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None # Ensure model is None if loading fails
        else:
            print(f"Failed to download model weights from {self.model_weights_url}. Model not loaded.")
            self.model = None

        os.makedirs(self.temp_image_dir, exist_ok=True)


    def infer(self, request: RequestObjects) -> ResponseObject:
        unique_filename = f"{uuid.uuid4()}.jpg"
        downloaded_image_path = os.path.join(self.temp_image_dir, unique_filename)
        download_file(request.image_url, downloaded_image_path, overwrite=True)
        print(f"Performing inference on {downloaded_image_path} with conf: {request.confidence_threshold}")
        results = self.model.predict(
            source=downloaded_image_path,
            conf=request.confidence_threshold,
            device=self.device)
      
        return_result = {"boxes": [], "confidences": [], "class_ids": [], "class_names": [], "annotated_image_base64_str": ""}
        for result in results:
            return_result["boxes"] = result.boxes.xyxyn.flatten().tolist()
            return_result["confidences"] = result.boxes.conf.tolist()
            return_result["class_ids"] = result.boxes.cls.tolist()
            return_result["class_names"] = [result.names[int(cls_id)] for cls_id in return_result["class_ids"]]

            annotated_image_np = result.plot()
            is_success, im_buf_arr = cv2.imencode(".jpg", annotated_image_np)
            byte_im = im_buf_arr.tobytes()
            return_result["annotated_image_base64_str"] = base64.b64encode(byte_im).decode('utf-8')
          
        os.remove(downloaded_image_path)  
        return ResponseObject(**return_result)
      
    def finalize(self):
        self.model = None
