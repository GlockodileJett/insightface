# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Content Safety Team
# @Time          : 2024-12-20
# @Function      : Content safety and NSFW detection utilities

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import warnings

class NSFWDetector:
    """
    NSFW (Not Safe For Work) content detector for images.
    This class provides methods to detect potentially inappropriate content
    and prevent the generation of NSFW images through face swapping.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize NSFW detector.
        
        Args:
            strict_mode: If True, uses more conservative filtering thresholds
        """
        self.strict_mode = strict_mode
        self.enabled = True
        
        # Define skin tone ranges in HSV color space for skin detection
        # Human skin typically has hue values between 0-20 and 160-180 (reddish/orange tones)
        # and relatively high saturation and value
        self.skin_lower1 = np.array([0, 30, 60], dtype=np.uint8)
        self.skin_upper1 = np.array([20, 255, 255], dtype=np.uint8)
        
        # Alternative skin tone range (for different lighting conditions)
        self.skin_lower2 = np.array([160, 30, 60], dtype=np.uint8) 
        self.skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    
    def disable(self):
        """Disable NSFW filtering (use with caution)"""
        self.enabled = False
        warnings.warn("NSFW filtering has been disabled. Use responsibly.", UserWarning)
    
    def enable(self):
        """Enable NSFW filtering"""
        self.enabled = True
    
    def _detect_skin_ratio(self, image: np.ndarray) -> float:
        """
        Detect the ratio of skin-colored pixels in the image using YCrCb color space.
        This is generally more reliable for skin detection than HSV.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Ratio of skin pixels (0.0 to 1.0)
        """
        if image is None or image.size == 0:
            return 0.0
            
        # Convert BGR to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        # These ranges are empirically determined for human skin tones
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask for skin color detection
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Additional HSV-based detection for better coverage
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # HSV ranges for skin tones
        lower_hsv1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv1 = np.array([20, 255, 255], dtype=np.uint8)
        
        lower_hsv2 = np.array([160, 20, 70], dtype=np.uint8)
        upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
        mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        
        # Combine both masks
        combined_mask = cv2.bitwise_or(mask, mask_hsv)
        
        # Calculate skin ratio
        skin_pixels = cv2.countNonZero(combined_mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        return skin_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image characteristics that might indicate inappropriate content.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary with analysis results
        """
        characteristics = {
            'skin_ratio': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'resolution': (0, 0),
            'aspect_ratio': 0.0
        }
        
        if image is None or image.size == 0:
            return characteristics
        
        # Calculate basic image properties
        height, width = image.shape[:2]
        characteristics['resolution'] = (width, height)
        characteristics['aspect_ratio'] = width / height if height > 0 else 0.0
        
        # Calculate skin ratio
        characteristics['skin_ratio'] = self._detect_skin_ratio(image)
        
        # Calculate brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        characteristics['brightness'] = np.mean(gray) / 255.0
        characteristics['contrast'] = np.std(gray) / 255.0
        
        return characteristics
    
    def is_potentially_nsfw(self, image: np.ndarray, face_bbox: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Check if an image is potentially NSFW based on various heuristics.
        
        Args:
            image: Input image in BGR format
            face_bbox: Optional face bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (is_nsfw: bool, reason: str)
        """
        if not self.enabled:
            return False, "NSFW filtering disabled"
        
        if image is None or image.size == 0:
            return True, "Invalid or empty image"
        
        # Analyze image characteristics
        chars = self._analyze_image_characteristics(image)
        
        # Define thresholds based on strict mode
        skin_threshold = 0.6 if self.strict_mode else 0.75
        min_resolution = (100, 100)
        
        # Check resolution (very small images might be suspicious)
        if chars['resolution'][0] < min_resolution[0] or chars['resolution'][1] < min_resolution[1]:
            return True, f"Image resolution too small: {chars['resolution']}"
        
        # Check skin ratio (high skin exposure might indicate inappropriate content)
        if chars['skin_ratio'] > skin_threshold:
            return True, f"High skin exposure detected: {chars['skin_ratio']:.2f} > {skin_threshold}"
        
        # Check if face region is disproportionately small compared to image
        if face_bbox is not None:
            face_area = (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1])
            image_area = chars['resolution'][0] * chars['resolution'][1]
            face_ratio = face_area / image_area if image_area > 0 else 0.0
            
            # If face is very small compared to image, might indicate full body image
            if face_ratio < 0.05 and chars['skin_ratio'] > 0.4:
                return True, f"Small face with high skin exposure: face_ratio={face_ratio:.3f}, skin_ratio={chars['skin_ratio']:.2f}"
        
        return False, "Content appears safe"
    
    def validate_swap_operation(self, source_image: np.ndarray, target_image: np.ndarray, 
                              source_face_bbox: Optional[np.ndarray] = None,
                              target_face_bbox: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Validate if a face swap operation should be allowed.
        
        Args:
            source_image: Source image containing the face to be swapped
            target_image: Target image where the face will be placed
            source_face_bbox: Bounding box of source face
            target_face_bbox: Bounding box of target face
            
        Returns:
            Tuple of (is_allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, "NSFW filtering disabled"
        
        # Check source image
        is_nsfw, reason = self.is_potentially_nsfw(source_image, source_face_bbox)
        if is_nsfw:
            return False, f"Source image flagged: {reason}"
        
        # Check target image
        is_nsfw, reason = self.is_potentially_nsfw(target_image, target_face_bbox)
        if is_nsfw:
            return False, f"Target image flagged: {reason}"
        
        return True, "Face swap operation approved"


class ContentSafetyError(Exception):
    """Exception raised when content safety checks fail"""
    pass


def create_nsfw_detector(strict_mode: bool = True) -> NSFWDetector:
    """
    Factory function to create an NSFW detector instance.
    
    Args:
        strict_mode: If True, uses more conservative filtering thresholds
        
    Returns:
        NSFWDetector instance
    """
    return NSFWDetector(strict_mode=strict_mode)