#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for NSFW content filtering functionality.
"""

import sys
import os

# Add paths to find our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python-package'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python-package/insightface/utils'))

import numpy as np
import cv2

def test_content_safety():
    """Test the content safety functionality."""
    print("Testing Content Safety Module...")
    
    try:
        # Import the module directly
        from content_safety import NSFWDetector, ContentSafetyError, create_nsfw_detector
        
        # Test 1: Basic initialization
        print("\n1. Testing detector initialization...")
        detector = NSFWDetector(strict_mode=True)
        print("✓ Strict mode detector created")
        
        detector_lenient = NSFWDetector(strict_mode=False)
        print("✓ Lenient mode detector created")
        
        # Test 2: Factory function
        print("\n2. Testing factory function...")
        detector_factory = create_nsfw_detector(strict_mode=True)
        print("✓ Factory function works")
        
        # Test 3: Create test images
        print("\n3. Creating test images...")
        
        # Safe image (low skin ratio)
        safe_image = np.zeros((200, 200, 3), dtype=np.uint8)
        safe_image[:, :] = [100, 50, 50]  # BGR background
        safe_image[0:50, 0:50] = [156, 138, 206]  # Small realistic skin area
        print("✓ Safe test image created")
        
        # High skin ratio image
        high_skin_image = np.zeros((200, 200, 3), dtype=np.uint8)
        high_skin_image[:, :] = [156, 138, 206]  # Mostly realistic skin color
        print("✓ High skin ratio test image created")
        
        # Small image
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        print("✓ Small test image created")
        
        # Test 4: Basic NSFW detection
        print("\n4. Testing NSFW detection...")
        
        # Test safe image
        is_nsfw, reason = detector.is_potentially_nsfw(safe_image)
        print(f"Safe image: NSFW={is_nsfw}, Reason='{reason}'")
        assert not is_nsfw, "Safe image should not be flagged"
        print("✓ Safe image correctly identified")
        
        # Test high skin image
        is_nsfw, reason = detector.is_potentially_nsfw(high_skin_image)
        print(f"High skin image: NSFW={is_nsfw}, Reason='{reason}'")
        assert is_nsfw, "High skin image should be flagged"
        print("✓ High skin image correctly flagged")
        
        # Test small image
        is_nsfw, reason = detector.is_potentially_nsfw(small_image)
        print(f"Small image: NSFW={is_nsfw}, Reason='{reason}'")
        assert is_nsfw, "Small image should be flagged"
        print("✓ Small image correctly flagged")
        
        # Test 5: Enable/disable functionality
        print("\n5. Testing enable/disable functionality...")
        
        detector.disable()
        is_nsfw, reason = detector.is_potentially_nsfw(high_skin_image)
        assert not is_nsfw, "Should not flag when disabled"
        print("✓ Detector correctly disabled")
        
        detector.enable()
        is_nsfw, reason = detector.is_potentially_nsfw(high_skin_image)
        assert is_nsfw, "Should flag when re-enabled"
        print("✓ Detector correctly re-enabled")
        
        # Test 6: Swap operation validation
        print("\n6. Testing swap operation validation...")
        
        is_allowed, reason = detector.validate_swap_operation(safe_image, safe_image)
        assert is_allowed, "Safe swap should be allowed"
        print("✓ Safe swap operation allowed")
        
        is_allowed, reason = detector.validate_swap_operation(high_skin_image, safe_image)
        assert not is_allowed, "Problematic swap should be blocked"
        print("✓ Problematic swap operation blocked")
        
        # Test 7: ContentSafetyError
        print("\n7. Testing ContentSafetyError...")
        
        try:
            raise ContentSafetyError("Test error")
        except ContentSafetyError as e:
            assert str(e) == "Test error"
            print("✓ ContentSafetyError works correctly")
        
        # Test 8: Strict vs lenient mode
        print("\n8. Testing strict vs lenient mode...")
        
        # Create a moderately problematic image
        moderate_image = np.zeros((200, 200, 3), dtype=np.uint8)
        moderate_image[:, :] = [100, 50, 50]  # Background
        moderate_image[0:120, 0:120] = [156, 138, 206]  # Moderate skin area (36% skin)
        
        strict_result, _ = detector.is_potentially_nsfw(moderate_image)
        lenient_result, _ = detector_lenient.is_potentially_nsfw(moderate_image)
        
        print(f"Moderate image - Strict: {strict_result}, Lenient: {lenient_result}")
        print("✓ Strict vs lenient mode tested")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("Content safety module is working correctly.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_analysis():
    """Test detailed image analysis functionality."""
    print("\nTesting detailed image analysis...")
    
    from content_safety import NSFWDetector
    detector = NSFWDetector()
    
    # Create a test image with known characteristics
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    test_image[:, :] = [128, 128, 128]  # Gray background
    test_image[0:100, 0:150] = [156, 138, 206]  # Realistic skin area
    
    # Analyze characteristics
    chars = detector._analyze_image_characteristics(test_image)
    
    print(f"Image characteristics:")
    print(f"  Resolution: {chars['resolution']}")
    print(f"  Aspect ratio: {chars['aspect_ratio']:.2f}")
    print(f"  Skin ratio: {chars['skin_ratio']:.3f}")
    print(f"  Brightness: {chars['brightness']:.3f}")
    print(f"  Contrast: {chars['contrast']:.3f}")
    
    # Expected values
    assert chars['resolution'] == (400, 300), f"Expected (400, 300), got {chars['resolution']}"
    assert abs(chars['aspect_ratio'] - 4/3) < 0.01, f"Expected ~1.33, got {chars['aspect_ratio']}"
    assert 0.1 < chars['skin_ratio'] < 0.15, f"Expected ~0.125, got {chars['skin_ratio']}"
    
    print("✓ Image analysis working correctly")

if __name__ == '__main__':
    print("Running Content Safety Tests")
    print("="*50)
    
    success = test_content_safety()
    
    if success:
        test_image_analysis()
        print("\nAll tests completed successfully! 🎉")
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)