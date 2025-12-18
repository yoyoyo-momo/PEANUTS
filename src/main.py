"""
YOLO cup detection + depth-based empty checking pipeline.
"""

import time
import argparse
import threading
import numpy as np
import cv2
from nxlib.context import NxLib
from nxlib.command import NxLibCommand
from nxlib.constants import *
from nxlib.item import NxLibItem
import nxlib.api as api
from nxlib import Camera
from ultralytics import YOLO

from detect_cups import detect_cups
from check_empty import check_multiple_cups
from Server import TCPServer


def transform_bbox_to_pointmap(bbox, color_image, point_map):
    """Transform bbox from color image to point map coordinates."""
    color_h, color_w = color_image.shape[:2]
    point_h, point_w = point_map.shape[:2]
    
    x1, y1, x2, y2 = bbox
    
    return [x1, y1, x2, y2]

    # Calculate offset to center point map in color image
    offset_x = (color_w - point_w) / 2
    offset_y = (color_h - point_h) / 2
    
    # Transform coordinates
    x1_pm = x1 - offset_x
    y1_pm = y1 - offset_y
    x2_pm = x2 - offset_x
    y2_pm = y2 - offset_y
    
    return [x1_pm, y1_pm, x2_pm, y2_pm]


def download_point_map(stereo_serial: str):
    """Download point map from NxLib."""
    root = NxLibItem()
    pm_item = root[ITM_IMAGES][ITM_POINT_MAP]
    
    with NxLibCommand(CMD_DOWNLOAD_IMAGES) as cmd:
        cmd.parameters()[ITM_CAMERAS].set_json(f'["{stereo_serial}"]')
        cmd.parameters()[ITM_IMAGES].set_json(f'["{ITM_POINT_MAP}"]')
        cmd.execute()
    
    width = pm_item[ITM_WIDTH].as_int()
    height = pm_item[ITM_HEIGHT].as_int()
    data = pm_item[ITM_DATA].as_binary()
    arr = np.frombuffer(data, dtype=np.float32)
    
    if arr.size != width * height * 3:
        raise RuntimeError("Point map size mismatch")
    
    return arr.reshape(height, width, 3)

def reshape_point_map(point_map):
    """ Reshape the point map array from (m x n x 3) to ((m*n) x 3). """
    return point_map.reshape(
        (point_map.shape[0] * point_map.shape[1]), point_map.shape[2])

def capture(stereo_serial, color_serial):
    """Capture color image and point map."""
    # Capture from both cameras
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS].set_json(f'["{stereo_serial}", "{color_serial}"]')
        cmd.execute()
    
    # Get color image via temp file
    root = NxLibItem()
    color_raw = root[ITM_CAMERAS][color_serial][ITM_IMAGES][ITM_RAW]
    # with NxLibCommand(CMD_SAVE_IMAGE) as cmd:
    #     cmd.parameters()[ITM_NODE] = color_raw.path
    #     cmd.parameters()[ITM_FILENAME] = "temp_color.png"
    #     cmd.execute()
    color_image = color_raw.get_binary_data()
    # print(color_image.shape)
    
    # Compute point map
    
    NxLibCommand(CMD_RECTIFY_IMAGES).execute()
    NxLibCommand(CMD_COMPUTE_DISPARITY_MAP).execute()
    NxLibCommand(CMD_COMPUTE_POINT_MAP).execute()
    # NxLibCommand(CMD_RENDER_POINT_MAP).execute()
    point_map = root[ITM_CAMERAS][stereo_serial][ITM_IMAGES][ITM_POINT_MAP]
    point_map_data = point_map.get_binary_data()

    return color_image, point_map_data


def main():
    parser = argparse.ArgumentParser(description="Cup detection and empty checking")
    parser.add_argument("--stereo", default="258076", help="Stereo camera serial")
    parser.add_argument("--color", default="4108896536", help="Color camera serial")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO confidence threshold")
    parser.add_argument("--table-depth", type=float, default=1500, help="Z coordinate of table surface (mm)")
    parser.add_argument("--empty-threshold", type=float, default=20.0, help="Max distance from table to consider empty (mm)")
    parser.add_argument("--interval", type=float, default=0.0, help="Interval between frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=unlimited)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--method", default="depth", choices=["depth", "yolo"], help="Method for empty checking")
    args = parser.parse_args()
    
    server = TCPServer(host="192.168.1.133", port=9999)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()

    print("Initializing NxLib...")
    try:
        api.initialize()
    except Exception as e:
        print(f"Failed to initialize NxLib: {e}")
        return
    
    try:
        with NxLib():
            print("NxLib context created")
            # Open cameras
            try:
                with NxLibCommand(CMD_OPEN) as cmd:
                    cmd.parameters()[ITM_CAMERAS].set_json(f'["{args.stereo}", "{args.color}"]')
                    cmd.execute()
                print("Cameras opened")
            except Exception as e:
                print(f"Failed to open cameras: {e}")
                return
            
            model = YOLO(args.model)

            frame_index = 0
            start_time = time.time()
            
            try:
                while True:
                    loop_start = time.time()
                    
                    # Capture images
                    try:
                        color_image, point_map = capture(args.stereo, args.color)
                    except Exception as e:
                        print(f"Capture failed: {e}")
                        break
                    
                    # Detect cups
                    try:
                        cup_detections = detect_cups(color_image, model, args.conf, device=args.device)
                    except Exception as e:
                        print(f"Detection failed: {e}")
                        break

                    if args.method == "depth":
                        bboxes_pm = [bbox for bbox, _, _ in cup_detections]
                        x1, y1, x2, y2 = map(int, bboxes_pm[0])
                        
                        # Check if cups are empty (returns list of (is_empty, center_xy) tuples)
                        empty_results = check_multiple_cups(point_map, bboxes_pm, args.table_depth, args.empty_threshold)
                        
                        # Draw results on image
                        elapsed = time.time() - loop_start
                        fps = 1.0 / elapsed if elapsed > 0 else 0
                        
                        display_image = color_image.copy()
                        
                        for cup_id, ((bbox, cls_id, conf), (is_empty, center_pm, rim_pm)) in enumerate(zip(cup_detections, empty_results), 1):
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Determine color and label
                            if is_empty:
                                # server.send_empty()
                                color = (0, 0, 255)  # Red for empty
                                label = f"Cup #{cup_id} EMPTY"
                            elif is_empty is False:
                                # server.send_not_empty()
                                color = (0, 255, 0)  # Green for not empty
                                label = f"Cup #{cup_id} NOT EMPTY"
                            else:
                                color = (0, 255, 255)  # Yellow for unknown
                                label = f"Cup #{cup_id} UNKNOWN"
                            
                            # server.send_not_empty()
                            server.send_empty()

                        #     # Draw bbox
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw rim points
                            if rim_pm is not None:
                                for px, py in rim_pm:
                                    cv2.circle(display_image, (px, py), 2, (255, 255, 0), -1)  # Cyan dots for rim
                            
                            # Draw rim center point
                            if center_pm is not None:
                                cv2.circle(display_image, (center_pm[0], center_pm[1]), 5, (255, 0, 255), -1)  # Magenta circle
                                cv2.circle(display_image, (center_pm[0], center_pm[1]), 7, (255, 255, 255), 2)  # White outline
                            
                            # Draw label
                            label_text = f"{label} ({conf:.2f})"
                            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                            cv2.putText(display_image, label_text, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    elif args.method == "yolo":
                        elapsed = time.time() - loop_start
                        fps = 1.0 / elapsed if elapsed > 0 else 0
                        display_image = color_image.copy()

                        for cup_id, (bbox, cls_id, conf) in enumerate(cup_detections, 1):
                            x1, y1, x2, y2 = map(int, bbox)

                            # Determine color and label
                            if cls_id == 0:
                                color = (0, 0, 255)  # Red for YOLO empty
                                label = f"Cup #{cup_id} Empty"
                            elif cls_id == 1:
                                color = (0, 255, 0)  # Green for YOLO not empty
                                label = f"Cup #{cup_id} not Empty"
                            elif cls_id == 2:
                                color = (255, 255, 0)  # Cyan for YOLO unknown
                                label = f"Cup #{cup_id} Unknown"
                            elif cls_id == 3:
                                color = (255, 0, 0)  # Blue for YOLO Small cup
                                label = f"Cup #{cup_id} Small"
                            
                            # Draw bbox
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label_text = f"{label} ({conf:.2f})"
                            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                            cv2.putText(display_image, label_text, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                        cls_ids = [cls_id for _, cls_id, _ in cup_detections]
                        print(cls_ids)
                        # if 0 in cls_ids:
                        #     server.send_empty()
                        # else:
                        #     server.send_not_empty()
                        
                        server.send_not_empty()
                        

                    # Draw FPS
                    fps_text = f"FPS: {fps:.2f} | Cups: {len(cup_detections)}"
                    cv2.putText(display_image, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Create point map visualization
                    # pm_viz = point_map[:, :, 2].copy()  # Z channel
                    # # Normalize for visualization
                    # pm_valid = pm_viz[~np.isnan(pm_viz) & (pm_viz > 0)]
                    # if pm_valid.size > 0:
                    #     pm_min, pm_max = np.percentile(pm_valid, [1, 99])
                    #     pm_viz = np.clip(pm_viz, pm_min, pm_max)
                    #     pm_viz = ((pm_viz - pm_min) / (pm_max - pm_min) * 255).astype(np.uint8)
                    #     pm_viz[np.isnan(point_map[:, :, 2]) | (point_map[:, :, 2] <= 0)] = 0
                    #     pm_viz = cv2.applyColorMap(pm_viz, cv2.COLORMAP_JET)
                        
                    #     # Draw center points on point map view
                    #     for (is_empty, center_pm, rim_pm) in empty_results:
                    #         if center_pm is not None:
                    #             cv2.circle(pm_viz, (center_pm[0], center_pm[1]), 8, (255, 255, 255), -1)
                    #             cv2.circle(pm_viz, (center_pm[0], center_pm[1]), 10, (0, 0, 0), 2)
                    # else:
                    #     pm_viz = np.zeros((point_map.shape[0], point_map.shape[1], 3), dtype=np.uint8)
                    
                    # Show both views with resizable windows
                    cv2.namedWindow("Cup Detection", cv2.WINDOW_NORMAL)
                    # cv2.namedWindow("Point Map View", cv2.WINDOW_NORMAL)
                    cv2.imshow("Cup Detection", display_image)
                    # cv2.imshow("Point Map View", pm_viz)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_index += 1
                    
                    if args.max_frames and frame_index >= args.max_frames:
                        break
                    
                    if args.interval > 0:
                        time.sleep(max(0, args.interval - (time.time() - loop_start)))
                        
            except KeyboardInterrupt:
                print("\nShutting down...")
            finally:
                cv2.destroyAllWindows()
            
            if frame_index > 0:
                total_time = time.time() - start_time
                print(f"Average FPS: {frame_index/total_time:.2f}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
