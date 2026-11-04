import pytest
import numpy as np
from vision_service.inference.volume import estimate_volume_cm3


def _create_synthetic_scene(h=100, w=100):
    """Helper to generate a valid, simple depth scene."""
    depth = np.ones((h, w), dtype=np.float32) * 1.0  # Background at 1m
    mask = np.zeros((h, w), dtype=bool)
    mask[30:70, 30:70] = True  # 40x40 = 1600 pixels (Food)
    depth[mask] = 0.8  # Food is closer (0.8m)
    return depth, mask, 100.0  # focal_length_px


def test_volume_happy_path():
    """Test that a valid synthetic scene calculates a positive volume."""
    depth, mask, focal = _create_synthetic_scene()
    
    result = estimate_volume_cm3(depth, mask, focal)
    
    assert "volume_cm3" in result
    assert result["volume_cm3"] > 0
    assert result["valid_volume_pixels"] > 0


def test_volume_shape_mismatch():
    """Test that mismatched image and mask dimensions fail safely."""
    depth = np.ones((100, 100))
    mask = np.ones((50, 50), dtype=bool)
    
    with pytest.raises(ValueError, match="Depth map and mask dimensions do not match"):
        estimate_volume_cm3(depth, mask, 100.0)


def test_volume_invalid_focal_length():
    """Test that zero or negative focal lengths are rejected."""
    depth, mask, _ = _create_synthetic_scene()
    
    with pytest.raises(ValueError, match="Focal length must be positive"):
        estimate_volume_cm3(depth, mask, 0)
        
    with pytest.raises(ValueError, match="Focal length must be positive"):
        estimate_volume_cm3(depth, mask, -50)


def test_volume_too_few_food_pixels():
    """Test that tiny masks (less than 50 pixels) are rejected."""
    depth, mask, focal = _create_synthetic_scene()
    
    # Shrink the mask to only 25 pixels
    small_mask = np.zeros_like(mask)
    small_mask[50:55, 50:55] = True 
    
    with pytest.raises(ValueError, match="Food segmentation contains too few valid pixels"):
        estimate_volume_cm3(depth, small_mask, focal)


def test_volume_invalid_depth_values():
    """Test that NaN depth values in the mask are rejected."""
    depth, mask, focal = _create_synthetic_scene()
    
    # Corrupt the food depth with NaNs
    depth[mask] = np.nan
    
    with pytest.raises(ValueError, match="Food segmentation contains too few valid pixels"):
        estimate_volume_cm3(depth, mask, focal)


def test_volume_no_background():
    """Test that if the food mask covers the whole image, it fails safely."""
    depth, _, focal = _create_synthetic_scene()
    
    # Mask the entire image
    full_mask = np.ones_like(depth, dtype=bool)
    
    with pytest.raises(ValueError, match="Not enough visible supporting-surface pixels"):
        estimate_volume_cm3(depth, full_mask, focal)


def test_volume_debug_mode():
    """Test that return_debug=True provides the expected debug dictionary."""
    depth, mask, focal = _create_synthetic_scene()
    
    result = estimate_volume_cm3(depth, mask, focal, return_debug=True)
    
    assert "_debug" in result
    assert "plane_normal" in result["_debug"]
    assert "height_max" in result["_debug"]
    assert result["_debug"]["height_max"] > 0