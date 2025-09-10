# Fisheye Camera Calibration Module

This module provides comprehensive fisheye camera calibration and distortion correction functionality for the Gesture Recognition System. It's designed to be modular, easy to use, and suitable for drone applications.

## Features

- **Fisheye Camera Calibration**: Accurate calibration using chessboard patterns
- **Real-time Distortion Correction**: Fast distortion correction with optimized algorithms
- **Interactive Calibration App**: User-friendly application for camera calibration
- **Parameter Adjustment**: Real-time parameter tuning for optimal results
- **Visualization Tools**: Comprehensive visualization for calibration process
- **Performance Monitoring**: Built-in performance statistics and quality assessment

## Module Structure

```
camera_calibration/
├── __init__.py                 # Module initialization
├── fisheye_calibrator.py       # Core calibration functionality
├── distortion_corrector.py     # Real-time distortion correction
├── calibration_visualizer.py   # Visualization tools
├── calibration_app.py          # Interactive calibration application
└── README.md                   # This documentation
```

## Quick Start

### 1. Basic Usage

```python
from modules.camera_calibration import (
    FisheyeCalibrator, 
    DistortionCorrector, 
    CalibrationVisualizer
)

# Initialize calibrator
calibrator = FisheyeCalibrator(chessboard_size=(9, 6), square_size=1.0)

# Add calibration images (with chessboard pattern)
for image in calibration_images:
    calibrator.add_calibration_image(image)

# Perform calibration
result = calibrator.calibrate()

# Create distortion corrector
corrector = DistortionCorrector(result)

# Correct images
corrected_image = corrector.correct_distortion(fisheye_image)
```

### 2. Interactive Calibration

Run the interactive calibration application:

```bash
cd modules/camera_calibration
python calibration_app.py --camera 0 --board-width 9 --board-height 6
```

**Controls:**
- `c`: Switch to capture mode
- `Space`: Capture calibration image
- `o`: Toggle auto-capture mode
- `l`: Perform calibration
- `t`: Test distortion correction
- `a`: Adjust correction parameters
- `s`: Save calibration data
- `d`: Load calibration data
- `h`: Toggle help display
- `q`: Quit application

### 3. Integration with Main System

```python
# In your main gesture recognition system
from modules.camera_calibration import DistortionCorrector

class GestureControlSystem:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize distortion corrector
        self.distortion_corrector = DistortionCorrector()
        
        # Load pre-calibrated parameters
        if self.load_camera_calibration():
            print("Fisheye correction enabled")
        else:
            print("Running without fisheye correction")
    
    def load_camera_calibration(self):
        try:
            from modules.camera_calibration import FisheyeCalibrator
            calibrator = FisheyeCalibrator()
            if calibrator.load_calibration("calibration_data/fisheye_calibration.json"):
                self.distortion_corrector.set_calibration(calibrator.calibration_result)
                return True
        except Exception as e:
            logger.warning(f"Failed to load camera calibration: {e}")
        return False
    
    def process_frame(self):
        # Get camera frame
        frame = self.camera_capture.get_frame()
        
        # Apply distortion correction if available
        if self.distortion_corrector.is_initialized:
            frame = self.distortion_corrector.correct_distortion(frame)
        
        # Continue with existing processing...
        # pose detection, gesture recognition, etc.
```

## Calibration Process

### 1. Preparation

1. **Print Chessboard Pattern**: Print a high-quality chessboard pattern
   - Default size: 9x6 internal corners
   - Ensure flat printing without distortion
   - Mount on rigid surface

2. **Setup Environment**: 
   - Good, even lighting
   - Stable camera mount
   - Clear view of chessboard

### 2. Calibration Steps

1. **Start Application**: `python calibration_app.py`
2. **Capture Mode**: Press `c` to enter capture mode
3. **Collect Images**: 
   - Hold chessboard at various positions and angles
   - Capture 15-30 images with `Space` or enable auto-capture with `o`
   - Cover different areas of the fisheye field of view
4. **Calibrate**: Press `l` to perform calibration
5. **Test Results**: Press `t` to test distortion correction
6. **Adjust Parameters**: Press `a` to fine-tune correction parameters
7. **Save**: Press `s` to save calibration data

### 3. Quality Assessment

Good calibration should have:
- **RMS Error < 1.0**: Lower is better
- **15+ Images**: More images improve accuracy
- **Good Coverage**: Images from different angles and distances
- **Clear Patterns**: All chessboard corners detected

## Parameter Tuning

### Distortion Correction Parameters

- **Balance (0.0-1.0)**: 
  - 0.0: Retain all original image content (black borders)
  - 1.0: Fill entire output image (may crop content)
  
- **FOV Scale (0.1-2.0)**: 
  - <1.0: Zoom in (crop field of view)
  - >1.0: Zoom out (expand field of view)
  
- **Crop Factor (0.1-1.0)**: 
  - Additional cropping of corrected image
  - Useful for removing distorted edges

### Real-time Adjustment

In adjust mode (`a`):
- `1/2`: Increase/decrease balance
- `3/4`: Increase/decrease FOV scale  
- `5/6`: Increase/decrease crop factor
- `r`: Reset to defaults

## Performance Optimization

### For Real-time Applications

```python
# Initialize once
corrector = DistortionCorrector(calibration_result)

# Fast correction using pre-computed maps
corrected = corrector.correct_distortion(image, fast_mode=True)
```

### For Drone Applications

```python
# Use adaptive corrector for varying conditions
from modules.camera_calibration import AdaptiveDistortionCorrector

corrector = AdaptiveDistortionCorrector(calibration_result)

# Auto-adjust parameters based on first few frames
corrector.auto_adjust_parameters(sample_image)

# Continue with optimized settings
corrected = corrector.correct_distortion(image)
```

## File Formats

### Calibration Data (JSON)
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coeffs": [k1, k2, k3, k4],
  "rms_error": 0.5,
  "image_size": [1280, 720],
  "calibration_date": "2024-01-01 12:00:00",
  "is_valid": true
}
```

## Troubleshooting

### Common Issues

1. **High RMS Error (>2.0)**:
   - Ensure chessboard is flat and high-quality
   - Improve lighting conditions
   - Capture more diverse images
   - Check camera focus

2. **Chessboard Not Detected**:
   - Improve lighting (avoid shadows/glare)
   - Ensure chessboard is clearly visible
   - Check chessboard size parameters
   - Clean camera lens

3. **Poor Correction Quality**:
   - Recalibrate with more images
   - Adjust correction parameters
   - Use higher resolution images
   - Ensure calibration covers full FOV

4. **Performance Issues**:
   - Use fast_mode=True for real-time
   - Reduce image resolution
   - Optimize correction parameters
   - Consider hardware acceleration

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration Examples

### Example 1: Gesture Recognition System

```python
def integrate_fisheye_correction(self):
    """Add fisheye correction to existing system"""
    try:
        from modules.camera_calibration import DistortionCorrector, FisheyeCalibrator
        
        # Load calibration
        calibrator = FisheyeCalibrator()
        if calibrator.load_calibration("calibration_data/fisheye_calibration.json"):
            self.fisheye_corrector = DistortionCorrector(calibrator.calibration_result)
            logger.info("Fisheye correction enabled")
            return True
    except Exception as e:
        logger.warning(f"Fisheye correction not available: {e}")
    
    self.fisheye_corrector = None
    return False

def process_camera_frame(self, frame):
    """Process frame with optional fisheye correction"""
    # Apply fisheye correction if available
    if self.fisheye_corrector and self.fisheye_corrector.is_initialized:
        frame = self.fisheye_corrector.correct_distortion(frame)
    
    # Continue with existing processing
    return self.existing_frame_processor(frame)
```

### Example 2: Parameter Adjustment UI

```python
def create_adjustment_interface(self):
    """Create parameter adjustment interface"""
    if not self.fisheye_corrector:
        return
    
    cv2.namedWindow("Fisheye Parameters")
    
    # Create trackbars
    cv2.createTrackbar("Balance", "Fisheye Parameters", 0, 100, self.on_balance_change)
    cv2.createTrackbar("FOV Scale", "Fisheye Parameters", 100, 200, self.on_fov_change)
    cv2.createTrackbar("Crop", "Fisheye Parameters", 100, 100, self.on_crop_change)

def on_balance_change(self, val):
    balance = val / 100.0
    self.fisheye_corrector.set_correction_parameters(balance=balance)

def on_fov_change(self, val):
    fov_scale = val / 100.0
    self.fisheye_corrector.set_correction_parameters(fov_scale=fov_scale)
```

## Future Enhancements

- **Multi-camera calibration**: Support for stereo fisheye setups
- **Online calibration**: Continuous calibration refinement
- **GPU acceleration**: CUDA/OpenCL support for faster processing
- **Automatic pattern detection**: Support for different calibration patterns
- **Quality-based parameter optimization**: Automatic parameter tuning

## License

This module is part of the Gesture Recognition Control System and follows the same license terms.
