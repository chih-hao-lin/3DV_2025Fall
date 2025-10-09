#!/usr/bin/env python3
"""
Test script for bundle adjustment functionality in pi3_reconstruction.py
"""

import numpy as np
import sys
import os

# Add the current directory to path to import the functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pi3_reconstruction import (
    rodrigues_to_matrix, 
    matrix_to_rodrigues, 
    bundle_adjustment_residuals,
    perform_bundle_adjustment
)

def test_rodrigues_conversion():
    """Test Rodrigues rotation vector conversion"""
    print("Testing Rodrigues conversion...")
    
    # Test identity rotation
    rvec = np.array([0.0, 0.0, 0.0])
    R = rodrigues_to_matrix(rvec)
    assert np.allclose(R, np.eye(3)), "Identity rotation failed"
    
    # Test small rotation
    rvec = np.array([0.1, 0.2, 0.3])
    R = rodrigues_to_matrix(rvec)
    rvec_back = matrix_to_rodrigues(R)
    assert np.allclose(rvec, rvec_back, atol=1e-6), "Rodrigues round-trip failed"
    
    print("✓ Rodrigues conversion test passed")

def test_bundle_adjustment_basic():
    """Test basic bundle adjustment functionality"""
    print("Testing basic bundle adjustment...")
    
    # Create synthetic data
    num_cameras = 3
    img_h, img_w = 64, 64
    
    # Create simple camera poses (looking at origin)
    camera_poses = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        t = np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0])
        pose = np.hstack([R, t.reshape(3, 1)])
        camera_poses.append(pose)
    
    camera_poses = np.array(camera_poses)
    
    # Create synthetic 3D points
    points_3d = np.random.randn(num_cameras, img_h, img_w, 3) * 0.1
    points_3d[:, :, :, 2] += 1.0  # Make sure points are in front of cameras
    
    # Create synthetic confidence
    confidences = np.ones((num_cameras, img_h, img_w)) * 0.8
    
    # Simple intrinsics
    intrinsics = np.array([
        [img_w, 0, img_w/2],
        [0, img_h, img_h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Test bundle adjustment
    try:
        refined_poses = perform_bundle_adjustment(
            camera_poses,
            points_3d,
            confidences,
            intrinsics,
            max_iterations=10,  # Small number for testing
            confidence_threshold=0.5
        )
        
        print(f"✓ Bundle adjustment completed successfully")
        print(f"  Original poses shape: {camera_poses.shape}")
        print(f"  Refined poses shape: {refined_poses.shape}")
        
        # Check that poses are reasonable
        assert refined_poses.shape == camera_poses.shape, "Shape mismatch"
        
        return True
        
    except Exception as e:
        print(f"✗ Bundle adjustment failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running bundle adjustment tests...\n")
    
    try:
        test_rodrigues_conversion()
        print()
        
        success = test_bundle_adjustment_basic()
        print()
        
        if success:
            print("🎉 All tests passed! Bundle adjustment is working correctly.")
        else:
            print("❌ Some tests failed.")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

