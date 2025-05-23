import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class DCE_Net(nn.Module):
    def __init__(self):
        super(DCE_Net, self).__init__()
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x_r = torch.tanh(self.conv7(x6))
        
        return x_r

class ZeroDCE:
    def __init__(self, device='cpu', weights_path='models/weights/Epoch99.pth'):
        self.device = device
        self.model = DCE_Net().to(device)
        self.load_weights(weights_path)
        
    def load_weights(self, weights_path):
        try:
            print(f"Loading weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            print("Initializing with default weights")
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            self.model.eval()
                        
    def enhance(self, img_tensor, alpha=0.1, gamma=1.2, use_nn=True):
        with torch.no_grad():
            if use_nn:
                print(f"Input tensor range: {img_tensor.min().item():.2f}-{img_tensor.max().item():.2f}")
                LE = self.model(img_tensor)
                print(f"Model output range: {LE.min().item():.2f}-{LE.max().item():.2f}")
                
                if LE.shape[1] != 24:
                    raise ValueError(f"Unexpected model output channels: {LE.shape[1]}, expected 24")
                
                LE_list = torch.split(LE, 3, dim=1)
                enhanced = img_tensor
                
                # Apply enhancement curves iteratively
                for i, le in enumerate(LE_list):
                    enhanced = enhanced + alpha * (enhanced - le * enhanced)
                    enhanced = torch.clamp(enhanced, 0.01, 1.0)  # Wider value range
                    print(f"Step {i+1} range: {enhanced.min().item():.2f}-{enhanced.max().item():.2f}")
                    
                # Stronger blending with original image
                enhanced = 0.5 * enhanced + 0.5 * img_tensor
            else:
                # Fallback to histogram adjustment
                enhanced = self.histogram_adjustment(img_tensor)
                
            enhanced = torch.clamp(enhanced, 0, 1)
            enhanced = torch.pow(enhanced, 1/gamma)
            print(f"Final enhanced tensor shape: {enhanced.shape}")
            return enhanced

    def histogram_adjustment(self, img_tensor):
        """Simple histogram-based exposure adjustment"""
        # Convert to numpy for processing
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Scale to 0-255 and convert to uint8
        img_scaled = np.clip(img_np * 255, 0, 255).astype('uint8')
        
        # Calculate histogram and cumulative distribution
        hist, bins = np.histogram(img_scaled.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        
        # Create lookup table
        lookup = np.interp(np.arange(0,256), bins[:-1], cdf_normalized)
        
        # Apply histogram equalization and scale back to 0-1
        enhanced_np = lookup[img_scaled].astype('float32') / 255.0
        
        print(f"Histogram adjustment - input range: {img_np.min()} {img_np.max()}")
        print(f"Enhanced range: {enhanced_np.min()} {enhanced_np.max()}")
        
        # Convert back to tensor
        enhanced = torch.from_numpy(enhanced_np).permute(2, 0, 1).unsqueeze(0)
        return enhanced.to(img_tensor.device)
            
    @staticmethod
    def preprocess(img_pil):
        img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        return img_tensor
        
    @staticmethod
    def postprocess(enhanced_tensor):
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print(f"Postprocess - enhanced range: {enhanced.min()} {enhanced.max()}")
        enhanced = np.clip(enhanced, 0, 1)  # Ensure valid range
        enhanced = (enhanced * 255).astype('uint8')
        print(f"Postprocess - final image range: {enhanced.min()} {enhanced.max()}")
        return Image.fromarray(enhanced)
