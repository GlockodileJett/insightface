import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface import ContentSafetyError

assert insightface.__version__>='0.7'

def safe_face_swap_example():
    """
    Example demonstrating safe face swapping with NSFW content filtering.
    """
    print("Initializing FaceAnalysis with content safety enabled...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize swapper with NSFW filtering enabled (default)
    print("Loading INSwapper model with content safety filtering...")
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    
    print("Processing test image...")
    img = ins_get_image('t1')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    
    if len(faces) < 2:
        print(f"Need at least 2 faces for swapping, found {len(faces)}")
        return
    
    print(f"Found {len(faces)} faces in the image")
    source_face = faces[0]  # Use first face as source
    
    # Demonstrate safe face swapping
    try:
        res = img.copy()
        swap_count = 0
        
        for i, face in enumerate(faces[1:], 1):  # Skip the source face
            try:
                print(f"Attempting to swap face {i}...")
                res = swapper.get(res, face, source_face, paste_back=True)
                swap_count += 1
                print(f"Successfully swapped face {i}")
            except ContentSafetyError as e:
                print(f"Face swap {i} blocked by content safety filter: {e}")
                continue
            except Exception as e:
                print(f"Face swap {i} failed due to technical error: {e}")
                continue
        
        if swap_count > 0:
            output_path = "./t1_safe_swapped.jpg"
            cv2.imwrite(output_path, res)
            print(f"Successfully completed {swap_count} face swaps. Output saved to {output_path}")
        else:
            print("No face swaps were completed due to safety restrictions or technical issues.")
    
    except ContentSafetyError as e:
        print(f"Face swap operation blocked by content safety filter: {e}")
        print("This is working as intended to prevent inappropriate content generation.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def demonstrate_nsfw_filtering():
    """
    Demonstrate NSFW filtering capabilities.
    """
    print("\n" + "="*60)
    print("CONTENT SAFETY DEMONSTRATION")
    print("="*60)
    
    from insightface.utils.content_safety import NSFWDetector
    
    # Create NSFW detector
    detector = NSFWDetector(strict_mode=True)
    
    print("Testing content safety detection on sample image...")
    
    # Test with the sample image
    img = ins_get_image('t1')
    is_nsfw, reason = detector.is_potentially_nsfw(img)
    
    print(f"Image analysis result: {'BLOCKED' if is_nsfw else 'APPROVED'}")
    print(f"Reason: {reason}")
    
    # Demonstrate how to disable filtering (not recommended for production)
    print("\nDemonstrating how to disable filtering (use with caution):")
    detector.disable()
    is_nsfw_disabled, reason_disabled = detector.is_potentially_nsfw(img)
    print(f"With filtering disabled: {'BLOCKED' if is_nsfw_disabled else 'APPROVED'}")
    print(f"Reason: {reason_disabled}")
    
    # Re-enable filtering
    detector.enable()
    print("Content safety filtering re-enabled.")

if __name__ == '__main__':
    print("Starting safe face swapping example with NSFW filtering...")
    print("This example demonstrates how InsightFace now includes content safety measures.")
    
    try:
        safe_face_swap_example()
        demonstrate_nsfw_filtering()
    except Exception as e:
        print(f"Example failed: {e}")
        print("This might be due to missing model files or dependencies.")
    
    print("\n" + "="*60)
    print("CONTENT SAFETY NOTES:")
    print("- NSFW filtering is enabled by default")
    print("- Use strict_mode=False for less conservative filtering")
    print("- Disable filtering only when appropriate and with proper oversight")
    print("- Consider implementing additional safety measures for production use")
    print("="*60)