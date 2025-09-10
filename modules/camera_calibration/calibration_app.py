"""
Fisheye Camera Calibration Application
Interactive application for fisheye camera calibration and testing
"""

import cv2
import numpy as np
import os
import argparse
import logging
from typing import Optional
from camera_calibration import (
    FisheyeCalibrator, 
    DistortionCorrector, 
    CalibrationVisualizer,
    CalibrationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationApp:
    """Interactive fisheye camera calibration application"""
    
    def __init__(self, 
                 camera_id: int = 0,
                 chessboard_size: tuple = (9, 6),
                 square_size: float = 1.0,
                 save_path: str = "calibration_data"):
        """
        Initialize calibration application
        
        Args:
            camera_id: Camera device ID
            chessboard_size: Chessboard pattern size (corners)
            square_size: Physical size of chessboard squares
            save_path: Directory to save calibration data
        """
        self.camera_id = camera_id
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.save_path = save_path
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize components
        self.calibrator = FisheyeCalibrator(chessboard_size, square_size)
        self.corrector = DistortionCorrector()
        self.visualizer = CalibrationVisualizer()
        
        # Application state
        self.camera = None
        self.mode = "capture"  # capture, calibrate, test, adjust
        self.target_images = 20
        self.auto_capture = False
        self.capture_delay = 2.0  # seconds between auto captures
        self.last_capture_time = 0
        
        # UI state
        self.show_help = True
        self.show_progress = True
        
        logger.info(f"Calibration app initialized for camera {camera_id}")
        logger.info(f"Chessboard: {chessboard_size}, Square size: {square_size}")
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def run(self):
        """Run the calibration application"""
        if not self.initialize_camera():
            return
        
        logger.info("Starting calibration application")
        logger.info("Press 'h' for help, 'q' to quit")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to read camera frame")
                    break
                
                # Process frame based on current mode
                if self.mode == "capture":
                    display_frame = self.process_capture_mode(frame)
                elif self.mode == "calibrate":
                    display_frame = self.process_calibrate_mode(frame)
                elif self.mode == "test":
                    display_frame = self.process_test_mode(frame)
                elif self.mode == "adjust":
                    display_frame = self.process_adjust_mode(frame)
                else:
                    display_frame = frame
                
                # Add UI overlays
                display_frame = self.add_ui_overlays(display_frame)
                
                # Show frame
                cv2.imshow("Fisheye Camera Calibration", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if self.handle_keyboard_input(key, frame):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self.cleanup()
    
    def process_capture_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in capture mode"""
        # Detect chessboard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Draw chessboard corners
        result = self.visualizer.draw_chessboard_corners(frame, corners, self.chessboard_size, found)
        
        # Auto capture if enabled
        if self.auto_capture and found:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.last_capture_time > self.capture_delay:
                if self.calibrator.add_calibration_image(frame):
                    self.last_capture_time = current_time
                    logger.info(f"Auto-captured image {len(self.calibrator.image_points)}")
        
        return result
    
    def process_calibrate_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in calibrate mode"""
        # Show calibration progress
        progress_panel = self.visualizer.draw_calibration_progress(self.calibrator, self.target_images)
        
        # Resize frame to fit with progress panel
        frame_resized = cv2.resize(frame, (frame.shape[1], frame.shape[0] - progress_panel.shape[0]))
        
        # Combine frame and progress panel
        result = np.vstack([frame_resized, progress_panel])
        return result
    
    def process_test_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in test mode"""
        if self.corrector.is_initialized:
            # Apply distortion correction
            corrected = self.corrector.correct_distortion(frame)
            if corrected is not None:
                return self.visualizer.create_distortion_comparison(frame, corrected, self.corrector)
        
        # Show message if no calibration available
        result = frame.copy()
        cv2.putText(result, "No calibration data available", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result, "Complete calibration first (mode 'c')", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return result
    
    def process_adjust_mode(self, frame: np.ndarray) -> np.ndarray:
        """Process frame in adjust mode"""
        if self.corrector.is_initialized:
            # Apply distortion correction
            corrected = self.corrector.correct_distortion(frame)
            if corrected is not None:
                # Create adjustment interface
                comparison = self.visualizer.create_distortion_comparison(frame, corrected, self.corrector)
                adjustment_panel = self.visualizer.create_parameter_adjustment_panel(self.corrector)
                
                # Combine views
                result = np.hstack([comparison, adjustment_panel])
                return result
        
        return frame
    
    def add_ui_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Add UI overlays to frame"""
        result = frame.copy()
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(result, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Help text
        if self.show_help:
            help_y = result.shape[0] - 120
            help_texts = [
                "Controls: 'c'-Capture, 'l'-Calibrate, 't'-Test, 'a'-Adjust",
                "'space'-Capture image, 'auto'-Toggle auto capture",
                "'s'-Save calibration, 'load'-Load calibration, 'h'-Toggle help"
            ]
            
            for i, text in enumerate(help_texts):
                cv2.putText(result, text, (10, help_y + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def handle_keyboard_input(self, key: int, frame: np.ndarray) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return True  # Quit
        
        elif key == ord('h'):
            self.show_help = not self.show_help
        
        elif key == ord('c'):
            self.mode = "capture"
            logger.info("Switched to capture mode")
        
        elif key == ord('l'):
            self.mode = "calibrate"
            self.perform_calibration()
        
        elif key == ord('t'):
            self.mode = "test"
            logger.info("Switched to test mode")
        
        elif key == ord('a'):
            self.mode = "adjust"
            logger.info("Switched to adjust mode")
        
        elif key == ord(' ') and self.mode == "capture":
            # Manual capture
            if self.calibrator.add_calibration_image(frame):
                logger.info(f"Captured image {len(self.calibrator.image_points)}")
        
        elif key == ord('o'):  # 'auto' - toggle auto capture
            self.auto_capture = not self.auto_capture
            logger.info(f"Auto capture: {'ON' if self.auto_capture else 'OFF'}")
        
        elif key == ord('s'):
            self.save_calibration()
        
        elif key == ord('d'):  # 'load'
            self.load_calibration()
        
        elif key == ord('r') and self.mode == "adjust":
            # Reset parameters to defaults
            self.corrector.set_correction_parameters(balance=0.0, fov_scale=1.0, crop_factor=1.0)
            logger.info("Reset correction parameters to defaults")
        
        # Parameter adjustment keys (adjust mode)
        elif self.mode == "adjust" and self.corrector.is_initialized:
            params = self.corrector.get_correction_parameters()
            
            if key == ord('1'):  # Increase balance
                new_balance = min(params['balance'] + 0.1, 1.0)
                self.corrector.set_correction_parameters(balance=new_balance)
            elif key == ord('2'):  # Decrease balance
                new_balance = max(params['balance'] - 0.1, 0.0)
                self.corrector.set_correction_parameters(balance=new_balance)
            elif key == ord('3'):  # Increase FOV scale
                new_fov = min(params['fov_scale'] + 0.1, 2.0)
                self.corrector.set_correction_parameters(fov_scale=new_fov)
            elif key == ord('4'):  # Decrease FOV scale
                new_fov = max(params['fov_scale'] - 0.1, 0.1)
                self.corrector.set_correction_parameters(fov_scale=new_fov)
            elif key == ord('5'):  # Increase crop factor
                new_crop = min(params['crop_factor'] + 0.1, 1.0)
                self.corrector.set_correction_parameters(crop_factor=new_crop)
            elif key == ord('6'):  # Decrease crop factor
                new_crop = max(params['crop_factor'] - 0.1, 0.1)
                self.corrector.set_correction_parameters(crop_factor=new_crop)
        
        return False
    
    def perform_calibration(self):
        """Perform camera calibration"""
        logger.info("Starting calibration process...")
        
        calibration_result = self.calibrator.calibrate()
        if calibration_result is not None:
            # Set calibration in corrector
            self.corrector.set_calibration(calibration_result)
            
            # Show calibration results
            logger.info("Calibration completed successfully!")
            logger.info(f"RMS Error: {calibration_result.rms_error:.4f}")
            logger.info(f"Images used: {len(self.calibrator.image_points)}")
            
            # Auto-save calibration
            self.save_calibration()
        else:
            logger.error("Calibration failed")
    
    def save_calibration(self):
        """Save calibration data"""
        if self.calibrator.calibration_result is None:
            logger.error("No calibration data to save")
            return
        
        filepath = os.path.join(self.save_path, "fisheye_calibration.json")
        if self.calibrator.save_calibration(filepath):
            logger.info(f"Calibration saved to {filepath}")
        else:
            logger.error("Failed to save calibration")
    
    def load_calibration(self):
        """Load calibration data"""
        filepath = os.path.join(self.save_path, "fisheye_calibration.json")
        if os.path.exists(filepath):
            if self.calibrator.load_calibration(filepath):
                self.corrector.set_calibration(self.calibrator.calibration_result)
                logger.info(f"Calibration loaded from {filepath}")
            else:
                logger.error("Failed to load calibration")
        else:
            logger.error(f"Calibration file not found: {filepath}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Application cleanup completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fisheye Camera Calibration Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--board-width", type=int, default=9, help="Chessboard width (corners)")
    parser.add_argument("--board-height", type=int, default=6, help="Chessboard height (corners)")
    parser.add_argument("--square-size", type=float, default=1.0, help="Chessboard square size")
    parser.add_argument("--save-path", type=str, default="calibration_data", help="Save directory")
    
    args = parser.parse_args()
    
    # Create and run application
    app = CalibrationApp(
        camera_id=args.camera,
        chessboard_size=(args.board_width, args.board_height),
        square_size=args.square_size,
        save_path=args.save_path
    )
    
    app.run()

if __name__ == "__main__":
    main()
