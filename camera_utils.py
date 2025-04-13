import cv2
import numpy as np
import time
import threading
import os
import string
import random

class CameraCapture:
    """
    A class to handle camera capture operations for the ASL recognition application.
    In cloud environments without camera access, it provides simulated ASL hand signs.
    """
    
    def __init__(self, source=0, width=640, height=480, force_real_camera=False):
        """
        Initialize the camera capture.
        
        Args:
            source: Camera source (0 for default webcam) or path to video file
            width: Frame width
            height: Frame height
            force_real_camera: If True, will not fall back to mock mode
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.stopped = False
        self.thread = None
        self.mock_mode = False
        self.current_letter_idx = 0
        self.letters = string.ascii_uppercase
        self.letter_change_time = time.time()
        self.letter_display_duration = 2.0  # Change letter every 2 seconds
        
        # Try to initialize real camera first
        try:
            self.initialize_camera()
            
            # Start the thread for reading frames
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            if force_real_camera:
                # Re-raise the exception if force_real_camera is True
                raise e
            else:
                print(f"Real camera initialization failed: {e}")
                print("Switching to mock camera mode...")
                self.mock_mode = True
                self.frame = self._create_mock_frame()
                
                # Start the thread for mock frame updates
                self.thread = threading.Thread(target=self._update_mock, args=())
                self.thread.daemon = True
                self.thread.start()
    
    def initialize_camera(self):
        """Initialize the camera or video capture."""
        try:
            # If source is a string, assume it's a video file path
            if isinstance(self.source, str):
                if not os.path.exists(self.source):
                    raise Exception(f"Video file not found: {self.source}")
                self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            # Set the frame width and height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise Exception("Error: Could not open camera or video source.")
            
            # Read the first frame
            ret, self.frame = self.cap.read()
            if not ret:
                raise Exception("Error: Could not read frame from camera or video source.")
            
        except Exception as e:
            raise Exception(f"Camera initialization error: {str(e)}")
    
    def _create_mock_frame(self):
        """Create a mock frame with an ASL hand sign image."""
        # Create a blank frame
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Get current letter to display
        if time.time() - self.letter_change_time > self.letter_display_duration:
            self.current_letter_idx = (self.current_letter_idx + 1) % len(self.letters)
            self.letter_change_time = time.time()
        
        current_letter = self.letters[self.current_letter_idx]
        
        # Draw hand shape region (simulated hand)
        hand_center = (self.width // 2, self.height // 2)
        hand_radius = min(self.width, self.height) // 4
        
        # Draw a circle representing the hand area
        cv2.circle(frame, hand_center, hand_radius, (220, 220, 220), -1)  # Filled circle for hand
        cv2.circle(frame, hand_center, hand_radius, (180, 180, 180), 2)   # Hand outline
        
        # Draw the letter in the center of the hand
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        text_size = cv2.getTextSize(current_letter, font, font_scale, 2)[0]
        text_x = hand_center[0] - text_size[0] // 2
        text_y = hand_center[1] + text_size[1] // 2
        
        # Draw the letter
        cv2.putText(frame, current_letter, (text_x, text_y), font, font_scale, (70, 70, 70), 3)
        
        # Add instructions
        cv2.putText(
            frame, 
            "Demo Mode: Simulating ASL Hand Signs", 
            (20, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 0), 
            2
        )
        
        # Add letter display
        cv2.putText(
            frame,
            f"Current ASL Letter: {current_letter}",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        return frame
    
    def _update(self):
        """
        Update the frame continuously in a separate thread for real camera.
        This helps ensure smooth frame capture.
        """
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    # If we've reached the end of a video file, we'll loop back to the beginning
                    if isinstance(self.source, str):  # If source is a file path
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                    else:
                        # Switch to mock mode if camera fails during operation
                        print("Camera failed during operation, switching to mock mode")
                        self.mock_mode = True
                        return
            time.sleep(0.01)  # Short sleep to reduce CPU usage
    
    def _update_mock(self):
        """Update the mock frame continuously in a separate thread."""
        while not self.stopped:
            self.frame = self._create_mock_frame()
            time.sleep(0.1)  # Update mock frame at 10 FPS
    
    def get_frame(self):
        """Get the current frame from the camera or mock source."""
        return self.frame
    
    def release(self):
        """Release the camera resources."""
        self.stopped = True
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if not self.mock_mode and self.cap is not None and self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.release()
        except:
            pass


class HandDetector:
    """
    A class to detect and track hands in frames.
    Uses OpenCV's built-in methods for simplicity.
    Also includes support for mock camera mode.
    """
    
    def __init__(self):
        """Initialize the hand detector."""
        # Parameters for skin detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    def detect_hand(self, frame):
        """
        Detect hand in the frame using skin color segmentation.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            ROI: Region of interest containing the hand
            hand_center: Center coordinates of the detected hand
            hand_radius: Approximate radius of the hand
        """
        # Check if this looks like a mock frame (detect the "Demo Mode" text)
        is_mock = self._check_if_mock_frame(frame)
        
        if is_mock:
            # For mock frames, we'll create a simulated hand ROI from the center area
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            radius = min(width, height) // 4
            
            # Extract a region around the center as the hand ROI
            margin = 20  # Add a margin around the hand circle
            x1 = max(0, center[0] - radius - margin)
            y1 = max(0, center[1] - radius - margin)
            x2 = min(width, center[0] + radius + margin)
            y2 = min(height, center[1] + radius + margin)
            
            roi = frame[y1:y2, x1:x2]
            
            return roi, center, radius
        
        # Regular hand detection for real camera frames
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask for skin color
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            
            # Apply morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assuming it's the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Get minimum enclosing circle for the contour
                ((center_x, center_y), radius) = cv2.minEnclosingCircle(max_contour)
                
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                # Return ROI and hand parameters
                if roi.size > 0:
                    return roi, (int(center_x), int(center_y)), int(radius)
        except Exception as e:
            print(f"Error in hand detection: {e}")
        
        # Return None if no hand detected or an error occurred
        return None, None, None
    
    def _check_if_mock_frame(self, frame):
        """
        Check if the frame is from our mock camera by looking for specific characteristics.
        
        Args:
            frame: Input frame
            
        Returns:
            True if it's a mock frame, False otherwise
        """
        # Convert to grayscale for text detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check if the frame has a consistent light gray background (mock frames have uniform backgrounds)
            height, width = gray.shape
            center_region = gray[height//4:3*height//4, width//4:3*width//4]
            std_dev = np.std(center_region)
            
            # Low standard deviation indicates uniform background (typical of our mock frames)
            is_uniform_background = std_dev < 25
            
            # Check if the frame has the specific light gray color of our mock frames
            avg_color = np.mean(center_region)
            is_expected_color = 200 < avg_color < 250
            
            return is_uniform_background and is_expected_color
        except:
            return False
    
    def draw_hand_markers(self, frame, center, radius):
        """
        Draw markers for the detected hand.
        
        Args:
            frame: Input frame
            center: Center coordinates of the detected hand
            radius: Approximate radius of the hand
            
        Returns:
            frame: Annotated frame
        """
        if center is not None and radius is not None:
            # Draw circle around the hand
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        return frame