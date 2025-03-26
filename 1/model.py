from clarifai.runners.models.model_class import ModelClass
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from typing import Iterator
import os
import torch
import sys
from pathlib import Path
import cv2
import numpy as np
import tempfile

# Add YOLOv5 modules to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Import YOLOv5 modules
try:
    from models.common import DetectMultiBackend
    from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
    from utils.torch_utils import select_device
    from utils.dataloaders import LoadImages
except ImportError:
    # Fallback paths for different directory structure
    YOLO_ROOT = Path('/app')  # Adjust this path based on your container structure
    if str(YOLO_ROOT) not in sys.path:
        sys.path.append(str(YOLO_ROOT))
    
    from models.common import DetectMultiBackend
    from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
    from utils.torch_utils import select_device
    from utils.dataloaders import LoadImages

class YourCustomModel(ModelClass):
    def load_model(self):
        '''Initialize and load the model here'''
        # Determine weights path based on environment
        if os.getenv("ENV_ID") == "dev":
            self.weights = "/weights/best.pt"
        else:
            self.weights = "/app/weights/best.pt"
        
        # Determine device based on environment
        if os.getenv("CUDA_VISIBLE_DEVICES") != "":
            self.device = "0"  # Use GPU
        else:
            self.device = "cpu"  # Use CPU
        
        # Model parameters
        self.imgsz = (640, 640)  # Inference size
        self.conf_thres = 0.3  # Confidence threshold
        self.iou_thres = 0.2  # NMS IoU threshold
        self.max_det = 1000  # Maximum detections per image
        self.classes = None  # Filter by class
        self.agnostic_nms = False  # Class-agnostic NMS
        self.augment = False  # Augmented inference
        self.half = False  # FP16 half-precision inference
        
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # Check image size
        
        # Warmup model
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))

    def predict(self, request: service_pb2.PostModelOutputsRequest
                ) -> Iterator[service_pb2.MultiOutputResponse]:
        # Process inputs and run model
        outputs = []
        
        for input_data in request.inputs:
            # Handle input image data
            if input_data.data.image.url:
                # For URL inputs, download and process the image
                # This is a simplified example - in production, you'd want to handle this more robustly
                import requests
                from io import BytesIO
                
                response = requests.get(input_data.data.image.url)
                img_bytes = BytesIO(response.content)
                
                # Save to a temporary file for processing
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img_bytes.getvalue())
                    img_path = tmp_file.name
            
            elif input_data.data.image.base64:
                # For base64 inputs
                import base64
                
                img_bytes = base64.b64decode(input_data.data.image.base64)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img_bytes)
                    img_path = tmp_file.name
            else:
                # Handle other input types if necessary
                continue
            
            # Load image
            dataset = LoadImages(img_path, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            
            # Process image
            regions = []
            for path, im, im0s, _, _ in dataset:
                # Prepare image
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                
                # Inference
                pred = self.model(im, augment=self.augment)
                
                # NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                           self.classes, self.agnostic_nms, max_det=self.max_det)
                
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                        
                        # Process detections
                        for *xyxy, conf, cls in reversed(det):
                            # Convert to Clarifai format
                            x1, y1, x2, y2 = xyxy
                            
                            # Calculate normalized coordinates
                            h, w = im0s.shape[:2]
                            top_row = float(y1) / h
                            left_col = float(x1) / w
                            bottom_row = float(y2) / h
                            right_col = float(x2) / w
                            
                            # Create region with bounding box
                            region = resources_pb2.Region(
                                region_info=resources_pb2.RegionInfo(
                                    bounding_box=resources_pb2.BoundingBox(
                                        top_row=top_row,
                                        left_col=left_col,
                                        bottom_row=bottom_row,
                                        right_col=right_col,
                                    ),
                                )
                            )
                            
                            # Add concept (class label)
                            class_id = int(cls)
                            class_name = self.names[class_id]
                            confidence = float(conf)
                            
                            concept = resources_pb2.Concept(
                                id=class_name,
                                name=class_name,
                                value=confidence
                            )
                            
                            # Add the concept to the region
                            region.data.concepts.append(concept)
                            regions.append(region)
            
            # Clean up temporary file
            try:
                os.unlink(img_path)
            except:
                pass
            
            # Create output with all regions
            output = resources_pb2.Output()
            output.data.regions.extend(regions)
            output.status.code = status_code_pb2.SUCCESS
            output.input.id = input_data.id
            
            outputs.append(output)
        
        # Return response with all outputs
        return service_pb2.MultiOutputResponse(
            outputs=outputs, 
            status=status_pb2.Status(code=status_code_pb2.SUCCESS)
        )

    def generate(self, request):
        '''Define streaming output logic if needed'''
        pass

    def stream(self, request):
        '''Handle both streaming input and output'''
        pass