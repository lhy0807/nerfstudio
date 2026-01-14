#!/usr/bin/env python3
"""
Convert Custom BRICS dataset to nerfstudio format.
Reads from processed data (videos/masks) and camera parameters from metric_params_refined_undistorted.txt.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
import h5py
import os

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    Uses COLMAP convention (scalar-first quaternions).
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def parse_params_txt(params_file: Path) -> List[Dict]:
    """
    Parse BRICS metric_params_refined_undistorted.txt file.
    
    Format: cam_id width height fx fy cx cy k1 k2 p1 p2 cam_name qvecw qvecx qvecy qvecz tvecx tvecy tvecz
    """
    cameras = []
    
    with open(params_file, 'r') as f:
        # Skip header line
        next(f)
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            # Ensure we have enough parts. The format implies at least 19 parts.
            if len(parts) < 19:
                continue
            
            cam_id = int(parts[0])
            width = int(parts[1])
            height = int(parts[2])
            fx = float(parts[3])
            fy = float(parts[4])
            cx = float(parts[5])
            cy = float(parts[6])
            k1 = float(parts[7])
            k2 = float(parts[8])
            p1 = float(parts[9])
            p2 = float(parts[10])
            cam_name = parts[11]
            qvecw = float(parts[12])
            qvecx = float(parts[13])
            qvecy = float(parts[14])
            qvecz = float(parts[15])
            tvecx = float(parts[16])
            tvecy = float(parts[17])
            tvecz = float(parts[18])
            
            cameras.append({
                'cam_id': cam_id,
                'width': width,
                'height': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'k1': k1,
                'k2': k2,
                'p1': p1,
                'p2': p2,
                'cam_name': cam_name,
                'qvec': np.array([qvecw, qvecx, qvecy, qvecz]),
                'tvec': np.array([tvecx, tvecy, tvecz]),
            })
    
    return cameras

def quaternion_translation_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert quaternion rotation and translation to camera-to-world transform matrix.
    """
    # Convert quaternion to rotation matrix
    rotation = qvec2rotmat(qvec)
    
    # Build world-to-camera matrix
    # COLMAP convention: tvec is translation from world to camera
    w2c = np.concatenate([rotation, tvec.reshape(3, 1)], axis=1)
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
    
    # Invert to get camera-to-world
    c2w = np.linalg.inv(w2c)
    
    # Convert from COLMAP/OpenCV convention to nerfstudio/OpenGL convention
    # Flip Y and Z axes
    c2w[0:3, 1:3] *= -1
    
    return c2w

def convert_custom_dataset(data_dir: Path, output_dir: Path, frame_idx: int = 0):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Locate params file
    params_file = data_dir / "metric_params_refined_undistorted.txt"
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found at {params_file}")
    
    print(f"Parsing parameters from {params_file}")
    cameras = parse_params_txt(params_file)
    print(f"Found {len(cameras)} cameras in params file")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    transforms = {
        "camera_model": "OPENCV",
        "ply_file_path": str(data_dir / "visual_hull_simple_full.ply"),
        "frames": []
    }
    
    for i, cam in enumerate(cameras):
        cam_name = cam['cam_name']
        print(f"Processing camera {i+1}/{len(cameras)}: {cam_name}")
        
        # 1. Get Extrinsics & Intrinsics
        c2w_opengl = quaternion_translation_to_c2w(cam['qvec'], cam['tvec'])
        
        # 2. Get Image (specific frame of video)
        video_path = data_dir / cam_name / "undistorted_cleaned.mp4"
        if not video_path.exists():
            video_path = data_dir / cam_name / "undistorted.mp4"
            if not video_path.exists():
                print(f"Warning: Video not found at {video_path} (or cleaned version), skipping.")
                continue
             
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")
            continue
        
        # 3. Get Mask (specific frame of h5) and combine as Alpha
        mask_path = data_dir / cam_name / "mask_refined.h5"
        
        # Prepare PIL Image
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        if mask_path.exists():
            try:
                with h5py.File(mask_path, 'r') as f:
                    # key 'data' shape (N, H, W)
                    mask_data = f['data'][frame_idx] # Get specific frame
                    
                # Normalize mask to 0-255 uint8
                if mask_data.max() <= 1.0:
                    mask_data = mask_data * 255.0
                mask_alpha = mask_data.clip(0, 255).astype(np.uint8)
                
                # Check size
                if mask_alpha.shape != frame_rgb.shape[:2]:
                    print(f"Warning: Mask shape {mask_alpha.shape} mismatch with image {frame_rgb.shape[:2]}")
                    # Alpha defaults to 255 (opaque) if mask mismatch
                    pil_image.putalpha(255)
                else:
                    pil_image.putalpha(Image.fromarray(mask_alpha))
                    
            except Exception as e:
                print(f"Error reading mask for {cam_name}: {e}")
                pil_image.putalpha(255)
        else:
            pil_image.putalpha(255)
        
        # Save Image (RGBA)
        image_filename = f"frame_{i:05d}.png"
        image_out_path = images_dir / image_filename
        pil_image.save(image_out_path)
        
        # 4. Add to transforms
        frame_entry = {
            "file_path": f"./images/{image_filename}",
            "transform_matrix": c2w_opengl.tolist(),
            "w": cam['width'],
            "h": cam['height'],
            "fl_x": cam['fx'],
            "fl_y": cam['fy'],
            "cx": cam['cx'],
            "cy": cam['cy'],
            "k1": cam['k1'],
            "k2": cam['k2'],
            "p1": cam['p1'],
            "p2": cam['p2'],
        }
            
        transforms["frames"].append(frame_entry)

    # Save transforms.json
    transforms_out_path = output_dir / "transforms.json"
    with open(transforms_out_path, 'w') as f:
        json.dump(transforms, f, indent=4)
        
    print(f"Conversion complete!")
    print(f"Saved {len(transforms['frames'])} frames to {output_dir}")
    print(f"Transforms: {transforms_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Custom BRICS dataset to structure.")
    parser.add_argument("--data_dir", required=True, help="Path to processed data (episode_0)")
    # Raw dir is no longer needed
    parser.add_argument("--output_dir", required=True, help="Path to output nerfstudio dataset")
    
    parser.add_argument("--frame_idx", type=int, default=0, help="Index of frame to extract (default=0)")
    args = parser.parse_args()
    
    convert_custom_dataset(args.data_dir, args.output_dir, args.frame_idx)

if __name__ == "__main__":
    main()
