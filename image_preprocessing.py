import cv2
import numpy as np
from camera_utils import HandDetector

def preprocess_frame(frame, target_size=(64, 64)):
    """
    Preprocess a video frame for ASL sign detection.
    
    Args:
        frame: Input video frame
        target_size: Target size for the processed image
        
    Returns:
        processed_img: Processed image ready for model input
        original_frame: Original frame with hand markers (for display)
    """
    if frame is None:
        # Return empty images if no frame is provided
        empty_img = np.zeros((*target_size, 3), dtype=np.uint8)
        return empty_img, empty_img
    
    # Make a copy of the original frame for drawing
    original_frame = frame.copy()
    
    # Create hand detector
    detector = HandDetector()
    
    # Detect hand in the frame
    hand_roi, hand_center, hand_radius = detector.detect_hand(frame)
    
    # Draw hand markers on the original frame
    if hand_center is not None and hand_radius is not None:
        original_frame = detector.draw_hand_markers(original_frame, hand_center, hand_radius)
    
    # Process the hand ROI if detected
    if hand_roi is not None and hand_roi.size > 0:
        # Convert to grayscale
        gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        resized_roi = cv2.resize(gray_roi, target_size)
        
        # Apply threshold to highlight hand silhouette
        _, thresh = cv2.threshold(resized_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Convert back to 3 channels for display
        processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return processed_img, original_frame
    else:
        # Return empty image if no hand detected
        empty_img = np.zeros((*target_size, 3), dtype=np.uint8)
        return empty_img, original_frame

def extract_hand_features(hand_img):
    """
    Extract features from a hand image for improved recognition.
    
    Args:
        hand_img: Preprocessed hand image
        
    Returns:
        features: Extracted features
    """
    # Convert to grayscale if needed
    if len(hand_img.shape) == 3:
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    # (Simple version, can be enhanced with actual HOG implementation)
    features = []
    
    # Simple gradient features
    gx = cv2.Sobel(hand_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(hand_img, cv2.CV_32F, 0, 1)
    
    # Gradient magnitude and direction
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Flatten and add to features
    features.extend(mag.flatten())
    
    return np.array(features)

def augment_hand_image(hand_img):
    """
    Apply data augmentation to a hand image to improve model generalization.
    
    Args:
        hand_img: Input hand image
        
    Returns:
        List of augmented images
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(hand_img)
    
    # Rotation
    for angle in [10, -10, 20, -20]:
        rows, cols = hand_img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(hand_img, rotation_matrix, (cols, rows))
        augmented_images.append(rotated)
    
    # Flipping
    flipped = cv2.flip(hand_img, 1)  # Horizontal flip
    augmented_images.append(flipped)
    
    # Brightness/contrast adjustments
    bright = cv2.convertScaleAbs(hand_img, alpha=1.2, beta=10)
    augmented_images.append(bright)
    
    dark = cv2.convertScaleAbs(hand_img, alpha=0.8, beta=-10)
    augmented_images.append(dark)
    
    return augmented_images