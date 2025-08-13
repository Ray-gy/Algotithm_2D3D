"""
演示脚本

展示第一阶段2D图像处理功能的基本用法。
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
from loguru import logger

from utils.file_io import load_config, ensure_directory
from core.image_processor import ImageProcessor
from core.segmentation import ToothSegmentation
from utils.visualization import ImageVisualizer

def main():
    """主演示函数"""
    
    # 配置日志
    logger.add("output/logs/demo.log", rotation="1 MB")
    logger.info("=== 牙齿图像处理演示程序启动 ===")
    
    try:
        # 1. 加载配置
        config_path = project_root / "config" / "image_processing.yaml"
        config = load_config(config_path)
        logger.info("配置文件加载完成")
        
        # 2. 初始化处理器
        image_processor = ImageProcessor(config)
        segmentation = ToothSegmentation(config)
        visualizer = ImageVisualizer(config)
        
        # 3. 查找测试图像
        sample_image_path = project_root / "20180101004543-H0000015.JPG"
        
        if not sample_image_path.exists():
            logger.error(f"测试图像不存在: {sample_image_path}")
            logger.info("请将显微镜图像放在项目根目录下")
            return
        
        # 4. 创建输出目录
        output_dir = ensure_directory(project_root / "output" / "demo")
        
        # 5. 图像处理流程演示
        logger.info(f"开始处理图像: {sample_image_path}")
        
        # 加载和预处理
        original_image = image_processor.load_image(str(sample_image_path))
        preprocessed_image = image_processor.preprocess_image(original_image)
        
        # 保存预处理结果
        cv2.imwrite(str(output_dir / "01_preprocessed.jpg"), preprocessed_image)
        
        # 图像分割
        segmented_image, mask = segmentation.segment_tooth(preprocessed_image, return_mask=True)
        
        # 保存分割结果
        cv2.imwrite(str(output_dir / "02_segmented.jpg"), segmented_image)
        cv2.imwrite(str(output_dir / "03_mask.jpg"), mask)
        
        # ROI提取
        roi_image, roi_mask, bbox = segmentation.crop_to_roi(segmented_image, mask)
        cv2.imwrite(str(output_dir / "04_roi.jpg"), roi_image)
        
        # 可视化对比
        comparison = visualizer.create_processing_comparison(
            original_image, preprocessed_image, segmented_image, roi_image
        )
        cv2.imwrite(str(output_dir / "05_comparison.jpg"), comparison)
        
        # 使用改进的Canny算法进行轮廓提取
        logger.info("=== 开始改进Canny算法轮廓提取 ===")
        _, contour_points, processing_info = segmentation.segment_and_extract_contours(
            original_image, use_advanced_canny=True
        )
        
        # 保存边缘检测结果
        if 'edges' in processing_info['contour_extraction']:
            edges = processing_info['contour_extraction']['edges']
            cv2.imwrite(str(output_dir / "06_edges.jpg"), edges)
        
        # 绘制轮廓点集可视化
        contour_visualization = original_image.copy()
        if len(contour_points) > 0:
            # 绘制轮廓点
            for point in contour_points:
                cv2.circle(contour_visualization, tuple(point.astype(int)), 1, (0, 0, 255), -1)
            
            # 绘制轮廓线
            if len(contour_points) > 2:
                pts = contour_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(contour_visualization, [pts], True, (255, 0, 0), 2)
        
        cv2.imwrite(str(output_dir / "07_contour_points.jpg"), contour_visualization)
        
        # 显示结果统计
        logger.info("=== 处理结果统计 ===")
        logger.info(f"原始图像尺寸: {original_image.shape}")
        logger.info(f"预处理后尺寸: {preprocessed_image.shape}")
        logger.info(f"分割掩码非零像素: {np.count_nonzero(mask)}")
        logger.info(f"ROI区域尺寸: {roi_image.shape}")
        logger.info(f"ROI边界框: {bbox}")
        
        # 改进Canny算法统计
        edge_info = processing_info['contour_extraction']['edge_detection']
        logger.info(f"边缘像素数: {edge_info['edge_pixels']}")
        logger.info(f"边缘比例: {edge_info['edge_ratio']:.3f}")
        logger.info(f"轮廓点数: {processing_info['contour_extraction']['contour_points_count']}")
        
        logger.info("=== 演示完成 ===")
        logger.info(f"结果已保存到: {output_dir}")
        logger.info("您可以查看以下文件:")
        logger.info("  - 01_preprocessed.jpg: 预处理结果")
        logger.info("  - 02_segmented.jpg: 分割结果") 
        logger.info("  - 03_mask.jpg: 分割掩码")
        logger.info("  - 04_roi.jpg: ROI区域")
        logger.info("  - 05_comparison.jpg: 处理对比图")
        logger.info("  - 06_edges.jpg: 改进Canny边缘检测结果")
        logger.info("  - 07_contour_points.jpg: 轮廓点集可视化")
        
    except Exception as e:
        logger.error(f"演示程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()