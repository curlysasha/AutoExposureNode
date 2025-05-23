import torch
import numpy as np
import cv2
from .models.zero_dce import ZeroDCE
import comfy.utils
from PIL import Image

class ZeroDCE_Node:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "models", "weights", "Epoch99.pth")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
            
        self.model = ZeroDCE(device=self.device, weights_path=weights_path)
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 0.3, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.2, "min": 0.8, "max": 1.5, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.12, "min": 0.0, "max": 2.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"

    def process(self, image, alpha, gamma, contrast, saturation):
        # Convert ComfyUI tensor to numpy
        image_np = image.cpu().numpy()[0]
        image_np = (image_np * 255).astype(np.uint8)
        
        # Process with ZeroDCE
        corrected = self.model.enhance(Image.fromarray(image_np), alpha=alpha, gamma=gamma)
        
        # Apply contrast
        corrected_np = np.array(corrected).astype(np.float32) / 255.0
        corrected_np = np.clip(corrected_np * contrast, 0.0, 1.0)
        
        # Apply saturation
        hsv = cv2.cvtColor(corrected_np, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        corrected_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to tensor
        corrected_tensor = torch.from_numpy(corrected_np).unsqueeze(0)
        
        return (corrected_tensor,)

NODE_CLASS_MAPPINGS = {
    "ZeroDCE_Node": ZeroDCE_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZeroDCE_Node": "Zero-DCE Exposure Correction"
}
