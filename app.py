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

@inferless.request
class RequestObjects(BaseModel):
    image_url: str = Field(default="https://github.com/rbgo404/Files/raw/main/photo-1517732306149-e8f829eb588a.jpeg")
    confidence_threshold: Optional[float] = Field(default=0.25)

@inferless.response
class ResponseObject(BaseModel):
    boxes: List[float] = []
    confidences: List[float] = []
    class_ids: List[float]= []
    class_names: List[str] = []
    annotated_image_base64_str: str = "Test output"

class InferlessPythonModel:
    @staticmethod
    def download_file(url: str, save_path: str, overwrite: bool = False) -> bool:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            return True
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(save_path, 'wb') as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                if total_size > 0:
                    done = int(50 * downloaded_size / total_size)
        return True

    def initialize(self, context=None):
        self.model_weights_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
        models_dir = os.path.join(os.getcwd(), "models_yolo")
        self.model_save_path = os.path.join(models_dir, "yolo11m.pt")
        self.temp_image_dir = os.path.join(os.getcwd(), "temp_images")

        if self.download_file(self.model_weights_url, self.model_save_path, overwrite=False):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.model_save_path)
            self.model.to(self.device) # Ensure model is on the correct device
        os.makedirs(self.temp_image_dir, exist_ok=True)

    def infer(self, request: RequestObjects) -> ResponseObject:
        unique_filename = f"{uuid.uuid4()}.jpg"
        downloaded_image_path = os.path.join(self.temp_image_dir, unique_filename)
        self.download_file(request.image_url, downloaded_image_path, overwrite=True)

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
