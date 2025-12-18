#! /usr/bin/env python3
# -*- coding: utf8 -*-

"""
Depth-based empty cup detection module.
Uses stereo depth data to determine if a cup is empty.
"""

import argparse
import numpy as np
import cv2
from typing import Optional, Dict, Any, List


def find_rim_and_center(point_map: np.ndarray, bbox: List[float], rim_percentile: float = 20.0, rim_band: float = 0.2, cup_id: int = 0) -> Optional[Dict[str, Any]]:
    """
    Find rim points by selecting the closest (smallest Z) depths in the bbox,
    then compute the center point of those rim points.

    Notes:
    - This version does NOT use any edge band; it considers the whole bbox.
    - `rim_band` is accepted for API compatibility but ignored.

    Args:
        point_map: (H, W, 3) array of XYZ coordinates (mm)
        bbox: [x1, y1, x2, y2] bounding box in pixels
        rim_percentile: Percentile of closest depths to treat as rim (default 10%)
        rim_band: Ignored (kept for signature compatibility)
        cup_id: Cup ID for debug messages (default 0)

    Returns:
        Dict with keys:
          - rim_z: median Z of rim points (mm)
          - center_z: median Z around the rim center point (mm)
          - center_xy: absolute pixel coords (x, y) of rim center
          - rim_points: list of absolute pixel coords [(x, y), ...]
    """

    return None

    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = point_map.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    depth_region = point_map[y1:y2, x1:x2, 2]
    valid_mask = ~np.isnan(depth_region) & (depth_region > 0)

    if valid_mask.sum() < 10:
        return None

    # Use percentile to pick "small depth" (closest to camera) points as rim
    valid_depths = depth_region[valid_mask]
    thresh = np.percentile(valid_depths, rim_percentile)
    rim_mask = valid_mask & (depth_region <= thresh)

    if rim_mask.sum() < 5:
        return None

    rim_depths = depth_region[rim_mask]
    rim_z = float(np.median(rim_depths))

    # Rim point coordinates in crop space (y, x), then convert to absolute
    rim_coords_crop = np.argwhere(rim_mask)
    rim_points = [(int(x1 + int(x)), int(y1 + int(y))) for y, x in rim_coords_crop]

    # Compute rim center as centroid of rim points in pixel space
    center_y_crop = int(np.mean(rim_coords_crop[:, 0]))
    center_x_crop = int(np.mean(rim_coords_crop[:, 1]))
    center_x = center_x_crop + x1
    center_y = center_y_crop + y1

    # Sample a small neighborhood around the center to get a stable center_z
    sample_radius = 5
    cy_min = max(y1, center_y - sample_radius)
    cy_max = min(y2, center_y + sample_radius + 1)
    cx_min = max(x1, center_x - sample_radius)
    cx_max = min(x2, center_x + sample_radius + 1)

    center_region = point_map[cy_min:cy_max, cx_min:cx_max, 2]
    center_valid = center_region[~np.isnan(center_region) & (center_region > 0)]
    if center_valid.size == 0:
        return None
    center_z = float(np.median(center_valid))

    # Report center pixel and its XYZ from point map
    if 0 <= center_y < h and 0 <= center_x < w:
        center_xyz = point_map[center_y, center_x, :]
        cup_prefix = f"Cup #{cup_id}: " if cup_id > 0 else ""
        print(f"{cup_prefix}Rim center pixel: ({center_x}, {center_y}), XYZ: ({center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f}) mm")

    return {
        "rim_z": rim_z,
        "center_z": center_z,
        "center_xy": (center_x, center_y),
        "rim_points": rim_points,
    }


def is_cup_empty(point_map: np.ndarray, bbox: List[float], table_depth: float, 
                 empty_threshold: float = 30.0, min_valid_ratio: float = 0.3, cup_id: int = 0) -> tuple:
    """
    Check if a cup is empty by finding the 3D center of the rim and sampling depth there.
    
    Args:
        point_map: (H, W, 3) array of XYZ coordinates
        bbox: [x1, y1, x2, y2] bounding box
        table_depth: Z coordinate of the table surface (mm)
        empty_threshold: Maximum distance (mm) from table to consider empty
        min_valid_ratio: Minimum ratio of valid depth points required
        cup_id: Cup ID for debug messages (default 0)
    
    Returns:
        Tuple of (is_empty, center_xy, rim_points)
    """
    # Find rim points
    rim_info = find_rim_and_center(point_map, bbox, cup_id=cup_id)
    if rim_info:
        # Get the rim points in pixel coordinates
        rim_points_px = rim_info["rim_points"]
        
        # Extract 3D XYZ positions of rim points
        rim_xyz = []
        for px, py in rim_points_px:
            if 0 <= py < point_map.shape[0] and 0 <= px < point_map.shape[1]:
                xyz = point_map[py, px, :]
                if not np.isnan(xyz[2]) and xyz[2] > 0:
                    rim_xyz.append(xyz)
        
        if len(rim_xyz) < 10:
            return (None, None, None)
        
        rim_xyz = np.array(rim_xyz)
        
        # Calculate 3D center of rim (mean of X, Y coordinates in world space)
        rim_center_x = np.mean(rim_xyz[:, 0])
        rim_center_y = np.mean(rim_xyz[:, 1])
        
        # Find the pixel closest to this 3D center point
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = point_map.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)
        
        bbox_region = point_map[y1:y2, x1:x2, :]
        
        # Calculate distance to rim center in XY plane
        distances = np.sqrt((bbox_region[:, :, 0] - rim_center_x)**2 + 
                           (bbox_region[:, :, 1] - rim_center_y)**2)
        
        # Mask out invalid points
        valid_mask = ~np.isnan(bbox_region[:, :, 2]) & (bbox_region[:, :, 2] > 0)
        distances[~valid_mask] = np.inf
        
        if not np.any(valid_mask):
            return (None, None, None)
        
        # Find pixel with minimum distance to rim center
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        center_depth = float(bbox_region[min_idx[0], min_idx[1], 2])
        
        # Convert to absolute pixel coords for reporting
        # center_px_abs = (x1 + min_idx[1], y1 + min_idx[0])
        # center_xyz_abs = point_map[center_px_abs[1], center_px_abs[0], :]
        cup_prefix = f"Cup #{cup_id}: " if cup_id > 0 else ""
        # print(f"{cup_prefix}Center pixel: ({center_px_abs[0]}, {center_px_abs[1]}), XYZ: ({center_xyz_abs[0]:.2f}, {center_xyz_abs[1]:.2f}, {center_xyz_abs[2]:.2f}) mm")
        
        # Empty cup: center depth close to table depth
        # Full cup: center depth significantly < table depth
        depth_diff = table_depth - center_depth
        is_empty = depth_diff < empty_threshold
        
        print(f"{cup_prefix}Rim center 3D: ({rim_center_x:.1f}, {rim_center_y:.1f}), Center Z: {center_depth:.2f}mm")
        print(f"{cup_prefix}Table Z: {table_depth:.2f}mm, Diff: {depth_diff:.2f}mm -> {'EMPTY' if is_empty else 'FULL'}")
        
        return (is_empty, rim_info["center_xy"], rim_info["rim_points"])
    
    # Fallback: sample center region directly
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = point_map.shape
    
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    # point map [y, x, 3]
    if x2 <= x1 or y2 <= y1:
        return (None, None, None)
    
    # Sample center 50% of bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = (x2 - x1) // 4
    half_h = (y2 - y1) // 4
    
    cx1 = max(x1, cx - half_w)
    cx2 = min(x2, cx + half_w)
    cy1 = max(y1, cy - half_h)
    cy2 = min(y2, cy + half_h)
    
    center_region = point_map[cy1:cy2, cx1:cx2, 2]
    valid_depths = center_region[~np.isnan(center_region) & (center_region > 0)]
    
    if valid_depths.size < 10:
        return (None, None, None)
    
    center_xy = ((x1 + x2) // 2, (y1 + y2) // 2)
    center_z = float(np.median(valid_depths))
    center_z = point_map[center_xy[1], center_xy[0], 2]
    depth_diff = abs(center_z - table_depth)
    # print(f"Cup #{cup_id}: Fallback center Z: {center_z:.2f}mm, Table Z: {table_depth:.2f}mm, Diff: {depth_diff:.2f}mm")
    
    # Return with center coordinates (no rim points for fallback)
    
    # Report fallback center pixel and XYZ
    center_xyz = point_map[center_xy[1], center_xy[0], :]
    cup_prefix = f"Cup #{cup_id}: " if cup_id > 0 else ""
    # print(f"{cup_prefix}Fallback center pixel: ({center_xy[0]}, {center_xy[1]}), XYZ: ({center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f}) mm")
    
    return (depth_diff < empty_threshold, center_xy, None)


def check_multiple_cups(point_map: np.ndarray, bboxes: List[List[float]], 
                        table_depth: float, empty_threshold: float = 30.0) -> List[tuple]:
    """
    Check multiple cups for empty status.
    
    Args:
        point_map: (H, W, 3) array of XYZ coordinates
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        table_depth: Z coordinate of the table surface (mm)
        empty_threshold: Maximum distance from table to consider empty (mm)
    
    Returns:
        List of (is_empty, center_xy, rim_points) tuples for each bbox
    """
    return [is_cup_empty(point_map, bbox, table_depth, empty_threshold, cup_id=i+1) for i, bbox in enumerate(bboxes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if cups are empty using depth data")
    parser.add_argument("--stereo", default="258076", help="Stereo camera serial")
    parser.add_argument("--color", default="4108896536", help="Color camera serial")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, 
                        metavar=("X1", "Y1", "X2", "Y2"), help="Bounding box coordinates")
    parser.add_argument("--table-depth", type=float, required=True, help="Z coordinate of table surface (mm)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Max distance from table to consider empty (mm)")
    args = parser.parse_args()
    
    # Import NxLib here to avoid dependency when using as module
    from nxlib.context import NxLib
    from nxlib.command import NxLibCommand
    from nxlib.constants import *
    from nxlib.item import NxLibItem
    import nxlib.api as api
    
    def download_point_map() -> np.ndarray:
        root = NxLibItem()
        pm_item = root[ITM_IMAGES][ITM_POINT_MAP]
        with NxLibCommand(CMD_DOWNLOAD_IMAGES) as cmd:
            cmd.parameters()[ITM_CAMERAS] = VAL_ALL
            cmd.parameters()[ITM_IMAGES].set_json(f'["{ITM_POINT_MAP}"]')
            cmd.execute()
            width = pm_item[ITM_WIDTH].as_int()
            height = pm_item[ITM_HEIGHT].as_int()
            data = pm_item[ITM_DATA].as_binary()
        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size != width * height * 3:
            raise RuntimeError("Point map size mismatch")
        return arr.reshape(height, width, 3)
    
    api.initialize()
    with NxLib():
        with NxLibCommand(CMD_OPEN) as cmd:
            cmd.parameters()[ITM_CAMERAS].set_json(f'["{args.stereo}", "{args.color}"]')
            cmd.execute()
        
        # Capture and compute depth
        with NxLibCommand(CMD_CAPTURE) as cmd:
            cmd.parameters()[ITM_CAMERAS].set_json(f'["{args.stereo}", "{args.color}"]')
            cmd.execute()
        NxLibCommand(CMD_RECTIFY_IMAGES).execute()
        NxLibCommand(CMD_COMPUTE_DISPARITY_MAP).execute()
        NxLibCommand(CMD_COMPUTE_POINT_MAP).execute()
        
        point_map = download_point_map()
        bbox = args.bbox
        is_empty, center_xy, rim_points = is_cup_empty(point_map, bbox, args.table_depth, args.threshold)
        
        if is_empty is None:
            print(f"Could not determine empty status for bbox {bbox}")
        else:
            print(f"Cup at {bbox}: {'EMPTY' if is_empty else 'NOT EMPTY'}")
            print(f"  Center: {center_xy}")
            print(f"  Rim points: {len(rim_points) if rim_points else 0}")
            print(f"  (Table depth: {args.table_depth}mm, Threshold: {args.threshold}mm)")
