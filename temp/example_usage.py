#!/usr/bin/env python3
"""
示例使用脚本
"""

from crop_image_patches import crop_image_to_patches
import os

def example_usage():
    """示例使用方法"""
    
    # 示例1: 使用custom_frames_grid.jpg（如果存在）
    sample_image = "../custom_frames_grid.jpg"
    
    if os.path.exists(sample_image):
        print("示例1: 使用 custom_frames_grid.jpg")
        print("=" * 40)
        crop_image_to_patches(sample_image, output_dir="./output")
        print("\n")
    else:
        print(f"示例图片不存在: {sample_image}")
    
    # 打印使用说明
    print("使用方法:")
    print("1. 命令行使用：")
    print("   python crop_image_patches.py 图片路径")
    print("   python crop_image_patches.py 图片路径 -o 输出目录")
    print("")
    print("2. 交互式使用：")
    print("   python crop_image_patches.py")
    print("   然后按提示输入图片路径和输出目录")
    print("")
    print("3. 在Python代码中使用：")
    print("   from crop_image_patches import crop_image_to_patches")
    print("   crop_image_to_patches('path/to/image.jpg', 'output_dir')")

if __name__ == "__main__":
    example_usage() 