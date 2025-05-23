import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class DCE_Net(nn.Module):
    def __init__(self):
        super(DCE_Net, self).__init__()
        self.e_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.e_conv7 = nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(x4))
        x6 = F.relu(self.e_conv6(x5))
        x7 = torch.tanh(self.e_conv7(x6))
        return x7

class ZeroDCE:
    def __init__(self, device='cpu', weights_path=None):
        self.device = device
        self.model = DCE_Net().to(device)
        self.weights_path = weights_path
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Initializing with random weights")

    def enhance(self, image, alpha=0.1, gamma=1.2):
        try:
            input_tensor = self.preprocess(image)
            with torch.no_grad():
                output_curves = self.model(input_tensor.to(self.device))
            return self.postprocess(input_tensor, output_curves, alpha, gamma)
        except Exception as e:
            print(f"Neural enhancement failed: {e}")
            print("Falling back to histogram adjustment")
            return self.histogram_adjustment(image)

    def histogram_adjustment(self, image):
        img = np.array(image)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))

    @staticmethod
    def preprocess(image):
        img = np.array(image).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img

    @staticmethod
    def postprocess(input_tensor, curves_tensor, alpha, gamma):
        # Convert tensors to numpy arrays
        input_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        curves = curves_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Reshape curves to (height, width, 8, 3)
        h, w, _ = curves.shape
        curves = curves.reshape(h, w, 8, 3)
        
        # Scale curves from [-1, 1] to [0, 1] using tanh output
        curves = (curves + 1) / 2  # Now in [0, 1]
        
        # Apply gamma correction to input
        enhanced = np.power(input_img.copy(), 1/gamma)  # Using inverse gamma for correction
        
        # Apply enhancement curves with safe scaling
        for i in range(8):
            enhanced += alpha * enhanced * curves[:, :, i, :]
            enhanced = np.clip(enhanced, 0, 2)  # Allow some overexposure
            
        # Normalize with epsilon to prevent division by zero
        eps = 1e-7
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + eps)
        
        # Convert to 8-bit with proper scaling
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        print(f"Debug - Curves range: {curves.min():.2f}-{curves.max():.2f}")
        print(f"Debug - Enhanced range pre-clip: {enhanced.min()}-{enhanced.max()}")
        
        return Image.fromarray(enhanced)
