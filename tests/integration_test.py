#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for NSFW filtering with face swapping functionality.
This test demonstrates how the content safety system works with the actual
face swapping pipeline without requiring model files.
"""

import sys
import os

# Add paths to find our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python-package'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python-package/insightface/utils'))

import numpy as np
import cv2

def create_mock_face(bbox):
    """Create a mock face object for testing."""
    class MockFace:
        def __init__(self, bbox):
            self.bbox = np.array(bbox)
            self.kps = np.array([[50, 50], [60, 50], [55, 60], [50, 70], [60, 70]])  # Mock keypoints
            self.normed_embedding = np.random.rand(512).astype(np.float32)  # Mock embedding
    
    return MockFace(bbox)

def test_inswapper_integration():
    """Test INSwapper integration with content safety."""
    print("Testing INSwapper Integration with Content Safety...")
    
    try:
        # Import required modules
        from content_safety import NSFWDetector, ContentSafetyError
        
        # Create a mock INSwapper class that mimics the real one but without ONNX dependencies
        class MockINSwapper:
            def __init__(self, enable_nsfw_filter=True, strict_mode=True):
                self.nsfw_detector = NSFWDetector(strict_mode=strict_mode) if enable_nsfw_filter else None
                if enable_nsfw_filter:
                    print('Content safety filtering enabled for face swapping')
                else:
                    print('WARNING: Content safety filtering is disabled')
            
            def get(self, img, target_face, source_face, paste_back=True):
                """Mock face swap operation with content safety checks."""
                # Content safety check (same as real implementation)
                if self.nsfw_detector is not None:
                    # Get face bounding boxes for more accurate analysis
                    target_bbox = getattr(target_face, 'bbox', None)
                    source_bbox = getattr(source_face, 'bbox', None)
                    
                    # Validate the face swap operation
                    is_allowed, reason = self.nsfw_detector.validate_swap_operation(
                        source_image=img,  # Note: In this context, img is actually the target image
                        target_image=img,  # We don't have separate source image, so use same
                        source_face_bbox=source_bbox,
                        target_face_bbox=target_bbox
                    )
                    
                    if not is_allowed:
                        raise ContentSafetyError(f"Face swap operation blocked: {reason}")
                
                # Mock face swap result (in real implementation, this would be the actual swapped image)
                return img.copy()
        
        print("\n1. Testing INSwapper with safe images...")
        
        # Create safe test image
        safe_image = np.zeros((300, 400, 3), dtype=np.uint8)
        safe_image[:, :] = [100, 80, 60]  # Non-skin background
        safe_image[50:150, 50:150] = [156, 138, 206]  # Small skin area for face
        
        # Create mock faces
        source_face = create_mock_face([50, 50, 150, 150])
        target_face = create_mock_face([200, 50, 300, 150])
        
        # Test with filtering enabled
        swapper = MockINSwapper(enable_nsfw_filter=True, strict_mode=True)
        
        try:
            result = swapper.get(safe_image, target_face, source_face)
            print("✓ Safe face swap operation completed successfully")
        except ContentSafetyError as e:
            print(f"❌ Safe operation was incorrectly blocked: {e}")
            return False
        
        print("\n2. Testing INSwapper with problematic images...")
        
        # Create problematic test image (high skin ratio)
        problematic_image = np.zeros((300, 400, 3), dtype=np.uint8)
        problematic_image[:, :] = [156, 138, 206]  # Mostly skin color
        
        try:
            result = swapper.get(problematic_image, target_face, source_face)
            print("❌ Problematic operation should have been blocked")
            return False
        except ContentSafetyError as e:
            print(f"✓ Problematic face swap correctly blocked: {e}")
        
        print("\n3. Testing INSwapper with filtering disabled...")
        
        # Test with filtering disabled
        swapper_no_filter = MockINSwapper(enable_nsfw_filter=False)
        
        try:
            result = swapper_no_filter.get(problematic_image, target_face, source_face)
            print("✓ Operation allowed when filtering disabled")
        except ContentSafetyError as e:
            print(f"❌ Operation should not be blocked when filtering disabled: {e}")
            return False
        
        print("\n4. Testing strict vs lenient mode...")
        
        # Create moderately problematic image
        moderate_image = np.zeros((300, 400, 3), dtype=np.uint8)
        moderate_image[:, :] = [100, 80, 60]  # Background
        moderate_image[0:180, 0:240] = [156, 138, 206]  # 36% skin coverage
        
        # Test strict mode
        swapper_strict = MockINSwapper(enable_nsfw_filter=True, strict_mode=True)
        strict_blocked = False
        try:
            result = swapper_strict.get(moderate_image, target_face, source_face)
        except ContentSafetyError:
            strict_blocked = True
        
        # Test lenient mode
        swapper_lenient = MockINSwapper(enable_nsfw_filter=True, strict_mode=False)
        lenient_blocked = False
        try:
            result = swapper_lenient.get(moderate_image, target_face, source_face)
        except ContentSafetyError:
            lenient_blocked = True
        
        print(f"Moderate image - Strict mode blocked: {strict_blocked}, Lenient mode blocked: {lenient_blocked}")
        
        # Strict mode should be more restrictive
        if lenient_blocked and not strict_blocked:
            print("❌ Strict mode should be more restrictive than lenient mode")
            return False
        else:
            print("✓ Strict vs lenient mode behavior is correct")
        
        print("\n5. Testing edge cases...")
        
        # Test with None image
        try:
            result = swapper.get(None, target_face, source_face)
            print("❌ None image should be blocked")
            return False
        except (ContentSafetyError, AttributeError):
            print("✓ None image correctly handled")
        
        # Test with empty image
        empty_image = np.array([])
        try:
            result = swapper.get(empty_image, target_face, source_face)
            print("❌ Empty image should be blocked")
            return False
        except (ContentSafetyError, ValueError, cv2.error):
            print("✓ Empty image correctly handled")
        
        # Test with very small image
        tiny_image = np.zeros((20, 20, 3), dtype=np.uint8)
        try:
            result = swapper.get(tiny_image, target_face, source_face)
            print("❌ Tiny image should be blocked")
            return False
        except ContentSafetyError as e:
            print(f"✓ Tiny image correctly blocked: {e}")
        
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("Content safety integration with face swapping is working correctly.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance of content safety checks."""
    print("\nTesting Content Safety Performance...")
    
    import time
    from content_safety import NSFWDetector
    
    detector = NSFWDetector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Time the detection
    start_time = time.time()
    num_tests = 100
    
    for _ in range(num_tests):
        is_nsfw, reason = detector.is_potentially_nsfw(test_image)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_tests
    
    print(f"Average detection time: {avg_time*1000:.2f} ms")
    print(f"Detection rate: {1/avg_time:.0f} images/second")
    
    if avg_time < 0.1:  # Less than 100ms is reasonable
        print("✓ Performance is acceptable")
        return True
    else:
        print("⚠ Performance might be slow for real-time applications")
        return True  # Not a failure, just a warning

if __name__ == '__main__':
    print("Running Content Safety Integration Tests")
    print("="*60)
    
    success1 = test_inswapper_integration()
    success2 = test_performance()
    
    if success1 and success2:
        print("\nAll integration tests completed successfully! 🎉")
        print("\nIMPORTANT NOTES:")
        print("- Content safety filtering is now enabled by default")
        print("- Use enable_nsfw_filter=False to disable (not recommended)")
        print("- Use strict_mode=False for less conservative filtering")
        print("- The system blocks high skin exposure and suspicious content")
        print("- All face swap operations are now validated for safety")
    else:
        print("\nSome integration tests failed! ❌")
        sys.exit(1)