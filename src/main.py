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


def capture(stereo_serial, color_serial):
    """Capture color image and point map."""
    # Capture from both cameras
    with NxLibCommand(CMD_CAPTURE) as cmd:
        cmd.parameters()[ITM_CAMERAS].set_json(f'["{stereo_serial}", "{color_serial}"]')
        cmd.execute()
    
    # Get color image raw
    root = NxLibItem()
    color_raw = root[ITM_CAMERAS][color_serial][ITM_IMAGES][ITM_RAW]
    color_image = color_raw.get_binary_data()
    
    # Compute point map
    
    NxLibCommand(CMD_RECTIFY_IMAGES).execute()
    NxLibCommand(CMD_COMPUTE_DISPARITY_MAP).execute()
    NxLibCommand(CMD_COMPUTE_POINT_MAP).execute()
    with NxLibCommand(CMD_RENDER_POINT_MAP) as cmd:
        cmd.parameters()[ITM_CAMERA] = f"{color_serial}"
        cmd.execute()
        point_map = cmd.result()[ITM_IMAGES][ITM_RENDER_POINT_MAP].get_binary_data()

    return color_image, point_map


def handle_depth_method(color_image, point_map, cup_detections, args, server):
    """Handle depth-based empty cup detection."""
    bboxes_pm = [bbox for bbox, _, _ in cup_detections]
    display_image = color_image.copy()
    
    if len(bboxes_pm) > 0:
        # Check if cups are empty
        empty_results = check_multiple_cups(point_map, bboxes_pm, args.table_depth, args.empty_threshold)
        
        # Track if any cup is empty
        any_cup_empty = False
        
        for cup_id, ((bbox, cls_id, conf), (is_empty, center_pm, rim_pm)) in enumerate(zip(cup_detections, empty_results), 1):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Determine color and label
            if is_empty:
                any_cup_empty = True
                color = (0, 0, 255)  # Red for empty
                label = f"Cup #{cup_id} EMPTY"
            elif is_empty is False:
                color = (0, 255, 0)  # Green for not empty
                label = f"Cup #{cup_id} NOT EMPTY"
            else:
                color = (0, 255, 255)  # Yellow for unknown
                label = f"Cup #{cup_id} UNKNOWN"
            
            # Draw bbox
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
        
        # Send message based on results
        if any_cup_empty:
            server.send_empty()
        else:
            server.send_not_empty()
    else:
        # No cups detected
        print("No cups detected")
        server.send_empty()
    
    return display_image


def handle_yolo_method(color_image, cup_detections, server):
    """Handle YOLO-based empty cup detection."""
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
    
    server.send_not_empty()
    
    return display_image


def main():
    parser = argparse.ArgumentParser(description="Cup detection and empty checking")
    parser.add_argument("--stereo", default="258076", help="Stereo camera serial")
    parser.add_argument("--color", default="4108896536", help="Color camera serial")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO confidence threshold")
    parser.add_argument("--table-depth", type=float, default=1500, help="Z coordinate of table surface (mm)")
    parser.add_argument("--empty-threshold", type=float, default=50.0, help="Max distance from table to consider empty (mm)")
    parser.add_argument("--interval", type=float, default=0.0, help="Interval between frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=unlimited)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--method", default="depth", choices=["depth", "yolo"], help="Method for empty checking")
    args = parser.parse_args()
    
    server = TCPServer(host="192.168.1.133", port=9090)
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

                    # Process based on method
                    elapsed = time.time() - loop_start
                    fps = 1.0 / elapsed if elapsed > 0 else 0
                    
                    if args.method == "depth":
                        display_image = handle_depth_method(color_image, point_map, cup_detections, args, server)
                    elif args.method == "yolo":
                        display_image = handle_yolo_method(color_image, cup_detections, server)

                    # Draw FPS
                    fps_text = f"FPS: {fps:.2f} | Cups: {len(cup_detections)}"
                    cv2.putText(display_image, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Show detection window
                    cv2.namedWindow("Cup Detection", cv2.WINDOW_NORMAL)
                    cv2.imshow("Cup Detection", display_image)
                    
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
