"""
调试轮廓提取问题
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from loguru import logger
from utils.file_io import load_config
from core.image_processor import ImageProcessor
from core.segmentation import ToothSegmentation

def debug_contours():
    """调试轮廓提取"""
    
    # 设置日志级别为DEBUG
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    try:
        # 1. 加载配置
        config_path = current_dir / "config" / "image_processing.yaml"
        config = load_config(config_path)
        
        # 2. 初始化处理器
        image_processor = ImageProcessor(config)
        segmentation = ToothSegmentation(config)
        
        # 3. 加载和处理图像
        sample_image_path = current_dir / "20180101004543-H0000015.JPG"
        original_image = image_processor.load_image(str(sample_image_path))
        
        # 4. 牙齿分割和ROI提取
        segmented_image, mask = segmentation.segment_tooth(original_image, return_mask=True)
        roi_image, roi_mask, bbox = segmentation.crop_to_roi(segmented_image, mask)
        
        # 5. 边缘检测
        edges, edge_info = segmentation.adaptive_canny.detect_edges(roi_image)
        
        print(f"\n=== 边缘检测结果 ===")
        print(f"ROI图像尺寸: {roi_image.shape}")
        print(f"边缘图像尺寸: {edges.shape}")
        print(f"边缘图像类型: {edges.dtype}")
        print(f"边缘像素数: {np.count_nonzero(edges)}")
        
        # 6. 直接使用OpenCV查找轮廓
        print(f"\n=== 直接轮廓检测 ===")
        edges_uint8 = edges.astype(np.uint8)
        contours, _ = cv2.findContours(edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"找到轮廓数量: {len(contours)}")
        
        # 显示每个轮廓的统计信息
        for i, contour in enumerate(contours[:10]):  # 只显示前10个
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            print(f"轮廓{i}: 面积={area:.2f}, 周长={perimeter:.2f}")
        
        # 7. 测试不同的过滤条件
        print(f"\n=== 测试不同过滤条件 ===")
        
        # 非常宽松的条件
        min_area_loose = 10
        max_area_loose = 1000000
        min_perimeter_loose = 5
        
        filtered_loose = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if min_area_loose <= area <= max_area_loose and perimeter >= min_perimeter_loose:
                filtered_loose.append(contour)
        
        print(f"宽松条件过滤后: {len(filtered_loose)}个轮廓")
        
        if filtered_loose:
            # 找到最大的轮廓
            largest_contour = max(filtered_loose, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            largest_perimeter = cv2.arcLength(largest_contour, True)
            points = largest_contour.reshape(-1, 2)
            
            print(f"最大轮廓: 面积={largest_area:.2f}, 周长={largest_perimeter:.2f}, 点数={len(points)}")
            
            # 保存可视化结果
            output_dir = current_dir / "output" / "debug"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存边缘图像
            cv2.imwrite(str(output_dir / "edges.jpg"), edges_uint8)
            
            # 绘制轮廓
            contour_vis = roi_image.copy()
            cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / "largest_contour.jpg"), contour_vis)
            
            print(f"结果保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"调试失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = debug_contours()
    if success:
        print("✅ 调试完成!")
    else:
        print("❌ 调试失败!")