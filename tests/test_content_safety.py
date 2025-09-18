#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NSFW content filtering functionality.
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add the parent directory to the path to import insightface
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python-package'))

try:
    from insightface.utils.content_safety import NSFWDetector, ContentSafetyError, create_nsfw_detector
except ImportError as e:
    print(f"Failed to import content safety module: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

class TestNSFWDetector(unittest.TestCase):
    """Test cases for NSFWDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = NSFWDetector(strict_mode=True)
        self.detector_lenient = NSFWDetector(strict_mode=False)
        
        # Create test images
        self.safe_image = self.create_test_image(skin_ratio=0.3)
        self.high_skin_image = self.create_test_image(skin_ratio=0.8)
        self.small_image = self.create_test_image(size=(50, 50))
        self.empty_image = np.array([])
        self.none_image = None
    
    def create_test_image(self, size=(200, 200), skin_ratio=0.3):
        """Create a test image with specified characteristics."""
        height, width = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill background with non-skin color (blue)
        image[:, :] = [100, 50, 50]  # BGR format
        
        # Add skin-colored regions
        if skin_ratio > 0:
            skin_pixels = int(height * width * skin_ratio)
            skin_height = int(np.sqrt(skin_pixels))
            skin_width = skin_pixels // skin_height
            
            # Create skin-colored rectangle
            end_y = min(height, skin_height)
            end_x = min(width, skin_width)
            image[0:end_y, 0:end_x] = [180, 150, 120]  # Skin-like color in BGR
        
        return image
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        # Test strict mode
        detector = NSFWDetector(strict_mode=True)
        self.assertTrue(detector.strict_mode)
        self.assertTrue(detector.enabled)
        
        # Test lenient mode
        detector = NSFWDetector(strict_mode=False)
        self.assertFalse(detector.strict_mode)
        self.assertTrue(detector.enabled)
    
    def test_enable_disable_filtering(self):
        """Test enabling and disabling filtering."""
        detector = NSFWDetector()
        
        # Initially enabled
        self.assertTrue(detector.enabled)
        
        # Disable
        detector.disable()
        self.assertFalse(detector.enabled)
        
        # Enable again
        detector.enable()
        self.assertTrue(detector.enabled)
    
    def test_skin_ratio_detection(self):
        """Test skin ratio detection functionality."""
        # Test safe image
        ratio = self.detector._detect_skin_ratio(self.safe_image)
        self.assertLess(ratio, 0.5, "Safe image should have low skin ratio")
        
        # Test high skin image
        ratio = self.detector._detect_skin_ratio(self.high_skin_image)
        self.assertGreater(ratio, 0.7, "High skin image should have high skin ratio")
        
        # Test empty image
        ratio = self.detector._detect_skin_ratio(self.empty_image)
        self.assertEqual(ratio, 0.0, "Empty image should have zero skin ratio")
        
        # Test None image
        ratio = self.detector._detect_skin_ratio(self.none_image)
        self.assertEqual(ratio, 0.0, "None image should have zero skin ratio")
    
    def test_image_characteristics_analysis(self):
        """Test image characteristics analysis."""
        chars = self.detector._analyze_image_characteristics(self.safe_image)
        
        # Check that all expected keys are present
        expected_keys = ['skin_ratio', 'brightness', 'contrast', 'resolution', 'aspect_ratio']
        for key in expected_keys:
            self.assertIn(key, chars, f"Missing key: {key}")
        
        # Check value ranges
        self.assertGreaterEqual(chars['skin_ratio'], 0.0)
        self.assertLessEqual(chars['skin_ratio'], 1.0)
        self.assertGreaterEqual(chars['brightness'], 0.0)
        self.assertLessEqual(chars['brightness'], 1.0)
        self.assertGreaterEqual(chars['contrast'], 0.0)
        self.assertLessEqual(chars['contrast'], 1.0)
        self.assertEqual(chars['resolution'], (200, 200))
        self.assertEqual(chars['aspect_ratio'], 1.0)
    
    def test_nsfw_detection_safe_image(self):
        """Test NSFW detection on safe image."""
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.safe_image)
        self.assertFalse(is_nsfw, f"Safe image should not be flagged as NSFW: {reason}")
    
    def test_nsfw_detection_high_skin_image(self):
        """Test NSFW detection on high skin ratio image."""
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.high_skin_image)
        self.assertTrue(is_nsfw, "High skin ratio image should be flagged as NSFW")
        self.assertIn("skin exposure", reason.lower(), "Reason should mention skin exposure")
    
    def test_nsfw_detection_small_image(self):
        """Test NSFW detection on small image."""
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.small_image)
        self.assertTrue(is_nsfw, "Small image should be flagged as NSFW")
        self.assertIn("resolution too small", reason.lower(), "Reason should mention small resolution")
    
    def test_nsfw_detection_invalid_images(self):
        """Test NSFW detection on invalid images."""
        # Test empty image
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.empty_image)
        self.assertTrue(is_nsfw, "Empty image should be flagged as NSFW")
        
        # Test None image
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.none_image)
        self.assertTrue(is_nsfw, "None image should be flagged as NSFW")
    
    def test_nsfw_detection_disabled(self):
        """Test NSFW detection when filtering is disabled."""
        self.detector.disable()
        
        # Even high skin ratio image should pass when filtering is disabled
        is_nsfw, reason = self.detector.is_potentially_nsfw(self.high_skin_image)
        self.assertFalse(is_nsfw, "No image should be flagged when filtering is disabled")
        self.assertIn("disabled", reason.lower(), "Reason should mention filtering is disabled")
    
    def test_strict_vs_lenient_mode(self):
        """Test difference between strict and lenient modes."""
        # Create a moderately high skin ratio image
        moderate_skin_image = self.create_test_image(skin_ratio=0.65)
        
        # Strict mode should flag it
        is_nsfw_strict, _ = self.detector.is_potentially_nsfw(moderate_skin_image)
        
        # Lenient mode might not flag it
        is_nsfw_lenient, _ = self.detector_lenient.is_potentially_nsfw(moderate_skin_image)
        
        # Strict mode should be more restrictive than lenient mode
        if is_nsfw_lenient:
            self.assertTrue(is_nsfw_strict, "If lenient mode flags it, strict mode should too")
    
    def test_validate_swap_operation(self):
        """Test swap operation validation."""
        # Test with safe images
        is_allowed, reason = self.detector.validate_swap_operation(
            self.safe_image, self.safe_image
        )
        self.assertTrue(is_allowed, f"Safe swap should be allowed: {reason}")
        
        # Test with problematic source
        is_allowed, reason = self.detector.validate_swap_operation(
            self.high_skin_image, self.safe_image
        )
        self.assertFalse(is_allowed, "Swap with problematic source should be blocked")
        self.assertIn("source", reason.lower(), "Reason should mention source image")
        
        # Test with problematic target
        is_allowed, reason = self.detector.validate_swap_operation(
            self.safe_image, self.high_skin_image
        )
        self.assertFalse(is_allowed, "Swap with problematic target should be blocked")
        self.assertIn("target", reason.lower(), "Reason should mention target image")
        
        # Test with filtering disabled
        self.detector.disable()
        is_allowed, reason = self.detector.validate_swap_operation(
            self.high_skin_image, self.high_skin_image
        )
        self.assertTrue(is_allowed, "Swap should be allowed when filtering is disabled")
    
    def test_content_safety_error(self):
        """Test ContentSafetyError exception."""
        try:
            raise ContentSafetyError("Test error message")
        except ContentSafetyError as e:
            self.assertEqual(str(e), "Test error message")
        except Exception:
            self.fail("ContentSafetyError should be raised correctly")
    
    def test_create_nsfw_detector_factory(self):
        """Test the factory function for creating NSFW detectors."""
        # Test strict mode
        detector = create_nsfw_detector(strict_mode=True)
        self.assertIsInstance(detector, NSFWDetector)
        self.assertTrue(detector.strict_mode)
        
        # Test lenient mode
        detector = create_nsfw_detector(strict_mode=False)
        self.assertIsInstance(detector, NSFWDetector)
        self.assertFalse(detector.strict_mode)


class TestIntegration(unittest.TestCase):
    """Integration tests for the NSFW filtering system."""
    
    def test_module_imports(self):
        """Test that all required modules can be imported."""
        try:
            from insightface.utils.content_safety import NSFWDetector, ContentSafetyError
            from insightface.model_zoo import ContentSafetyError as ModelZooContentSafetyError
            from insightface import ContentSafetyError as PackageContentSafetyError
            
            # All should be the same class
            self.assertEqual(ContentSafetyError, ModelZooContentSafetyError)
            self.assertEqual(ContentSafetyError, PackageContentSafetyError)
            
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")


def run_tests():
    """Run all tests."""
    print("Running NSFW detector tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNSFWDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\nAll tests passed! ({result.testsRun} tests)")
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        for test, traceback in result.failures + result.errors:
            print(f"FAILED: {test}")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)