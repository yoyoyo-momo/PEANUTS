#! /usr/bin/env python3
# -*- coding: utf8 -*-
"""
自動收集訓練數據 - 定時拍照保存
"""

import os
import time
import argparse
from datetime import datetime
from nxlib.context import NxLib, MonoCamera
from nxlib.command import NxLibCommand
from nxlib.constants import *
from nxlib.item import NxLibItem
import nxlib.api as api


def save_image(filename, item):
    with NxLibCommand(CMD_SAVE_IMAGE) as cmd:
        cmd.parameters()[ITM_NODE] = item.path
        cmd.parameters()[ITM_FILENAME] = filename
        cmd.execute()


def capture_dataset(output_dir, interval=5, max_images=10, stereo_serial="258076", color_serial="4108896536"):
    os.makedirs(output_dir, exist_ok=True)
    
    api.initialize()
    image_count = 0
    
    print(f"Starting data collection...")
    print(f"Output: {output_dir}")
    print(f"Interval: {interval}s, Max images: {max_images}")
    print(f"\n提示：在拍照期間請：")
    print("  - 改變杯子位置和數量")
    print("  - 嘗試不同填充狀態（空、半滿、全滿）")
    print("  - 改變光照條件")
    print("  - 添加遮擋物（手、紙巾等）")
    print("  - 調整相機角度（如果可能）\n")
    
    with NxLib(), MonoCamera(color_serial) as color_camera:
        # 如果不是 remote 模式，打開相機
        # if not os.getenv("NXLIB_REMOTE"):
        #     print("Opening cameras...")
        #     with NxLibCommand(CMD_OPEN) as cmd:
        #         cmd.parameters()[ITM_CAMERAS].set_json(f'["{color_serial}"]')
        #         cmd.execute()
        
        root = NxLibItem()
        
        try:
            while image_count < max_images:
                print(f"\rCapturing image {image_count + 1}/{max_images}...", end="", flush=True)
                
                # 捕獲
                # with NxLibCommand(CMD_CAPTURE) as cmd:
                #     cmd.parameters()[ITM_CAMERAS].set_json(f'["{color_serial}"]')
                #     cmd.execute()
                
                # # 處理渲染
                # NxLibCommand(CMD_RECTIFY_IMAGES).execute()
                # NxLibCommand(CMD_COMPUTE_DISPARITY_MAP).execute()
                # NxLibCommand(CMD_COMPUTE_POINT_MAP).execute()
                # NxLibCommand(CMD_RENDER_POINT_MAP).execute()

                color_camera.capture()
                color_camera.rectify()
                
                # 保存帶時間戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"img_{timestamp}.png")
                
                # texture_item = root[ITM_IMAGES][ITM_RENDER_POINT_MAP_TEXTURE]
                save_image(filename, color_camera[ITM_IMAGES][ITM_RAW])
                
                image_count += 1
                print(f" ✓ Saved: {filename}")
                
                if image_count < max_images:
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            print(f"\n\nInterrupted. Collected {image_count} images.")
        
        print(f"\n✓ Data collection complete: {image_count} images saved to {output_dir}")
        print(f"\n下一步：")
        print(f"  1. 使用標註工具標註圖片")
        print(f"  2. 轉換為 YOLO 格式")
        print(f"  3. 訓練模型")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動收集訓練數據集")
    parser.add_argument("--output", default="dataset/raw_images", help="輸出目錄")
    parser.add_argument("--interval", type=int, default=2, help="拍照間隔（秒）")
    parser.add_argument("--max-images", type=int, default=2000, help="最大圖片數量")
    parser.add_argument("--stereo", default="258076", help="Stereo 相機序號")
    parser.add_argument("--color", default="4108896536", help="Color 相機序號")
    args = parser.parse_args()
    
    capture_dataset(args.output, args.interval, args.max_images, args.stereo, args.color)
