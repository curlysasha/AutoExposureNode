# Neural Exposure Correction Tool

Web application and ComfyUI node for automatic image exposure correction using neural networks.

## Features

- Neural network-based exposure correction (Zero-DCE model)
- Adjustable parameters:
  - Alpha: Correction strength (0.05-0.3) Best 0.15
  - Gamma: Tonal curve adjustment (0.8-1.5) Best 1.2 + Contrast 1.2 + Saturation 1.12
- Auto-contrast fallback mode
- Web interface with real-time preview
- ComfyUI integration as custom node

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Flask
- Pillow
- OpenCV
- numpy

```bash
pip install flask pillow torch opencv-python numpy
```

## Usage

### Web Application
1. Start the server:
```bash
python app.py
```
2. Open `http://localhost:5000` in browser
3. Upload image and adjust parameters

### ComfyUI Node
1. Copy these files to ComfyUI's `custom_nodes` folder:
```
MyNode/
  __init__.py
  comfy_node.py
  models/
    __init__.py
    zero_dce.py
    weights/
      Epoch99.pth
```
2. Restart ComfyUI
3. Node will appear under "image/postprocessing"

## Folder Structure
```
.
├── app.py                 # Flask server
├── comfy_node.py          # ComfyUI integration
├── models/
│   ├── zero_dce.py        # Neural network model
│   └── weights/           # Pretrained weights
├── templates/             # Web interface
└── README.md
```

## Troubleshooting
- **Black output images**: Check model weights path and file permissions
- **Import errors**: Verify folder structure and __init__.py files
- **Slider not working**: Full reload the web page

## License
MIT License
