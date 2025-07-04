#!/usr/bin/env python3
"""
临时脚本：将图片裁成NxN的patches
输出：
1. 原图片带裁剪线
2. N*N个独立的patch文件
"""

import os
import sys
from PIL import Image, ImageDraw
import argparse

def crop_image_to_patches(input_path, grid_size=5, output_dir=None):
    """
    将图片裁成NxN的patches
    
    Args:
        input_path: 输入图片路径
        grid_size: 网格大小（NxN），默认为5
        output_dir: 输出目录，如果为None则使用输入图片所在目录
    """
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 {input_path}")
        return
    
    # 打开图片
    img = Image.open(input_path)
    
    # 如果图片有透明通道(RGBA)，转换为RGB模式
    if img.mode == 'RGBA':
        # 创建白色背景
        background = Image.new('RGB', img.size, (255, 255, 255))
        # 将RGBA图片合成到白色背景上
        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
        img = background
    elif img.mode != 'RGB':
        # 确保图片是RGB模式
        img = img.convert('RGB')
    
    width, height = img.size
    
    # 计算每个patch的尺寸
    patch_width = width // grid_size
    patch_height = height // grid_size
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    print(f"原图尺寸: {width} x {height}")
    print(f"网格大小: {grid_size} x {grid_size}")
    print(f"每个patch尺寸: {patch_width} x {patch_height}")
    
    # 1. 创建带裁剪线的原图
    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)
    
    # 绘制垂直线
    for i in range(1, grid_size):
        x = i * patch_width
        draw.line([(x, 0), (x, height)], fill='red', width=3)
    
    # 绘制水平线
    for i in range(1, grid_size):
        y = i * patch_height
        draw.line([(0, y), (width, y)], fill='red', width=3)
    
    # 保存带裁剪线的图片
    grid_output_path = os.path.join(output_dir, f"{base_name}_with_grid.jpg")
    img_with_lines.save(grid_output_path)
    print(f"保存带裁剪线的图片: {grid_output_path}")
    
    # 2. 裁剪并保存patches
    patch_paths = []
    total_patches = grid_size * grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            # 计算裁剪区域
            left = col * patch_width
            top = row * patch_height
            right = left + patch_width
            bottom = top + patch_height
            
            # 裁剪patch
            patch = img.crop((left, top, right, bottom))
            
            # 保存patch
            patch_filename = f"{base_name}_patch_{row}_{col}.jpg"
            patch_path = os.path.join(output_dir, patch_filename)
            patch.save(patch_path)
            patch_paths.append(patch_path)
            
            print(f"保存patch ({row},{col}): {patch_path}")
    
    print(f"\n完成！总共保存了{len(patch_paths)}个patches ({grid_size}x{grid_size})")
    print(f"输出目录: {output_dir}")
    
    return grid_output_path, patch_paths

def main():
    parser = argparse.ArgumentParser(description='将图片裁成NxN的patches')
    parser.add_argument('input_image', help='输入图片路径')
    parser.add_argument('-s', '--size', type=int, default=5, help='网格大小 (NxN), 默认为5')
    parser.add_argument('-o', '--output', help='输出目录', default=None)
    
    args = parser.parse_args()
    
    try:
        crop_image_to_patches(args.input_image, args.size, args.output)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式输入
    if len(sys.argv) == 1:
        print("图片裁剪工具")
        print("-" * 30)
        
        input_path = input("请输入图片路径: ").strip()
        if not input_path:
            print("未输入图片路径，退出。")
            sys.exit(1)
        
        # 输入网格大小
        grid_size_input = input("请输入网格大小 (例如: 3表示3x3, 回车默认5x5): ").strip()
        if grid_size_input:
            try:
                grid_size = int(grid_size_input)
                if grid_size <= 0:
                    print("网格大小必须大于0，使用默认值5")
                    grid_size = 5
            except ValueError:
                print("输入无效，使用默认值5")
                grid_size = 5
        else:
            grid_size = 5
        
        output_dir = input("请输入输出目录 (回车使用图片所在目录): ").strip()
        if not output_dir:
            output_dir = None
        
        crop_image_to_patches(input_path, grid_size, output_dir)
    else:
        main() 