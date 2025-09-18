# Content Safety and NSFW Filtering

InsightFace now includes comprehensive content safety measures to prevent the creation of inappropriate or NSFW (Not Safe For Work) content through face swapping. This document describes the content safety features and how to use them.

## Overview

The content safety system is designed to:
- Automatically detect potentially inappropriate content
- Block face swapping operations that might generate NSFW images
- Provide configurable filtering levels (strict vs lenient)
- Maintain user safety while preserving legitimate use cases

## Key Features

### 1. Automatic NSFW Detection
- **Skin exposure analysis**: Detects images with high ratios of skin-colored pixels
- **Resolution validation**: Blocks suspiciously small images
- **Content characteristics**: Analyzes brightness, contrast, and aspect ratios
- **Face-to-image ratio**: Checks if faces are proportionate to the overall image

### 2. Content Safety Integration
- **Enabled by default**: All face swapping operations are automatically filtered
- **Pre-operation validation**: Images are checked before any processing begins
- **Real-time blocking**: Inappropriate operations are stopped immediately
- **Detailed error messages**: Clear explanations when content is blocked

### 3. Configurable Filtering Levels
- **Strict mode** (default): Conservative filtering for maximum safety
- **Lenient mode**: Less restrictive filtering for creative applications
- **Disable option**: Complete bypass for controlled environments (use with caution)

## Quick Start

### Basic Usage with Default Settings

```python
import insightface
from insightface.app import FaceAnalysis

# Initialize with content safety enabled (default)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load face swapper with default safety settings
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)

# Face swapping operations are automatically filtered
try:
    result = swapper.get(target_image, target_face, source_face)
    # Success - content passed safety checks
except insightface.ContentSafetyError as e:
    print(f"Content blocked: {e}")
    # Handle blocked content appropriately
```

### Custom Configuration

```python
import insightface
from insightface.model_zoo import INSwapper

# Enable lenient mode (less restrictive)
swapper = INSwapper(
    model_file='inswapper_128.onnx',
    enable_nsfw_filter=True,
    strict_mode=False  # Use lenient filtering
)

# Disable filtering (not recommended for production)
swapper_no_filter = INSwapper(
    model_file='inswapper_128.onnx',
    enable_nsfw_filter=False  # Disable all filtering
)
```

### Standalone Content Analysis

```python
from insightface.utils.content_safety import NSFWDetector

# Create detector
detector = NSFWDetector(strict_mode=True)

# Analyze image
is_nsfw, reason = detector.is_potentially_nsfw(image)
if is_nsfw:
    print(f"Content flagged: {reason}")

# Validate face swap operation
is_allowed, reason = detector.validate_swap_operation(
    source_image=source_img,
    target_image=target_img
)
```

## Configuration Options

### NSFWDetector Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strict_mode` | bool | `True` | Use conservative filtering thresholds |

### INSwapper Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_nsfw_filter` | bool | `True` | Enable content safety filtering |
| `strict_mode` | bool | `True` | Use strict filtering rules |

## Detection Criteria

### Images That May Be Blocked

1. **High skin exposure**: Images with >60% (strict) or >75% (lenient) skin-colored pixels
2. **Low resolution**: Images smaller than 100x100 pixels
3. **Suspicious proportions**: Very small faces in images with high skin ratios
4. **Invalid content**: Empty, corrupted, or malformed images

### Images That Are Typically Allowed

1. **Portrait photos**: Standard headshots and face photos
2. **Group photos**: Multiple people in normal settings
3. **Professional images**: Business photos, headshots, etc.
4. **Creative content**: Art, illustrations (within reasonable bounds)

## Error Handling

### ContentSafetyError

When content is blocked, a `ContentSafetyError` is raised with a descriptive message:

```python
from insightface import ContentSafetyError

try:
    result = swapper.get(image, target_face, source_face)
except ContentSafetyError as e:
    # Handle blocked content
    print(f"Operation blocked: {e}")
    # Possible reasons:
    # - "High skin exposure detected: 0.85 > 0.6"
    # - "Image resolution too small: (50, 50)"
    # - "Small face with high skin exposure"
```

## Examples

### Example 1: Basic Face Swapping with Safety

```python
import cv2
import insightface
from insightface.app import FaceAnalysis

def main():
    # Initialize face analysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load swapper with safety enabled
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx')
    
    # Load images
    target_img = cv2.imread('target.jpg')
    source_img = cv2.imread('source.jpg')
    
    # Detect faces
    target_faces = app.get(target_img)
    source_faces = app.get(source_img)
    
    if target_faces and source_faces:
        try:
            # Perform safe face swap
            result = swapper.get(target_img, target_faces[0], source_faces[0])
            cv2.imwrite('result.jpg', result)
            print("Face swap completed successfully!")
        except insightface.ContentSafetyError as e:
            print(f"Face swap blocked: {e}")
    else:
        print("No faces detected in images")

if __name__ == '__main__':
    main()
```

## Support and Updates

For questions, issues, or feature requests related to content safety:

1. Check this documentation for common solutions
2. Review the test files for usage examples
3. Open an issue on the InsightFace GitHub repository
4. Contact the development team for enterprise support

---

**Note**: This content safety system provides technical safeguards but does not guarantee complete prevention of all inappropriate content. Users and developers should implement additional measures as needed for their specific applications and compliance requirements.