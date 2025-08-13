"""
测试改进的Canny边缘检测算法

用于验证自适应高斯滤波和改进Canny算法的功能
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加src目录到Python路径
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from loguru import logger
from utils.file_io import load_config, ensure_directory
from core.image_processor import ImageProcessor
from core.segmentation import ToothSegmentation


def test_advanced_canny():
    """测试改进的Canny算法"""
    
    # 配置日志
    logger.add("output/logs/test_canny.log", rotation="1 MB")
    logger.info("=== 开始测试改进Canny算法 ===")
    
    try:
        # 1. 加载配置
        config_path = current_dir / "config" / "image_processing.yaml"
        config = load_config(config_path)
        logger.info("配置文件加载完成")
        
        # 2. 初始化处理器
        image_processor = ImageProcessor(config)
        segmentation = ToothSegmentation(config)
        
        # 3. 加载测试图像
        sample_image_path = current_dir / "20180101004543-H0000015.JPG"
        
        if not sample_image_path.exists():
            logger.error(f"测试图像不存在: {sample_image_path}")
            return
        
        # 4. 创建输出目录
        output_dir = ensure_directory(current_dir / "output" / "canny_test")
        
        # 5. 处理图像
        logger.info(f"开始处理图像: {sample_image_path}")
        
        # 加载和预处理
        original_image = image_processor.load_image(str(sample_image_path))
        preprocessed_image = image_processor.preprocess_image(original_image)
        
        # 牙齿分割
        segmented_image, mask = segmentation.segment_tooth(preprocessed_image, return_mask=True)
        roi_image, _, _ = segmentation.crop_to_roi(segmented_image, mask)
        
        # 6. 测试改进的Canny算法
        logger.info("=== 测试改进Canny算法 ===")
        
        # 使用改进Canny算法
        contour_points, detection_info = segmentation.extract_contours_advanced_canny(roi_image)
        
        # 传统Canny算法对比
        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
        traditional_edges = cv2.Canny(gray_roi, 50, 150)
        
        # 7. 保存结果
        logger.info("保存处理结果...")
        
        # 保存原始和处理的图像
        cv2.imwrite(str(output_dir / "01_original.jpg"), original_image)
        cv2.imwrite(str(output_dir / "02_roi.jpg"), roi_image)
        
        # 保存边缘检测结果
        advanced_edges = detection_info['edges']
        cv2.imwrite(str(output_dir / "03_advanced_canny.jpg"), advanced_edges)
        cv2.imwrite(str(output_dir / "04_traditional_canny.jpg"), traditional_edges)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('改进Canny算法 vs 传统Canny算法', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始ROI图像')
        axes[0, 0].axis('off')
        
        # 灰度图像
        axes[0, 1].imshow(gray_roi, cmap='gray')
        axes[0, 1].set_title('灰度图像')
        axes[0, 1].axis('off')
        
        # 传统Canny
        axes[0, 2].imshow(traditional_edges, cmap='gray')
        axes[0, 2].set_title('传统Canny算法')
        axes[0, 2].axis('off')
        
        # 改进Canny
        axes[1, 0].imshow(advanced_edges, cmap='gray')
        axes[1, 0].set_title('改进Canny算法')
        axes[1, 0].axis('off')
        
        # 轮廓点可视化
        contour_vis = roi_image.copy()
        if len(contour_points) > 0:
            for point in contour_points[::5]:  # 每5个点绘制一个，减少密度
                cv2.circle(contour_vis, tuple(point.astype(int)), 2, (0, 0, 255), -1)
        
        axes[1, 1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'轮廓点({len(contour_points)}个)')
        axes[1, 1].axis('off')
        
        # 统计对比
        traditional_edge_count = np.count_nonzero(traditional_edges)
        advanced_edge_count = detection_info['edge_detection']['edge_pixels']
        
        stats_text = f"""边缘检测统计:
传统Canny: {traditional_edge_count} 像素
改进Canny: {advanced_edge_count} 像素
轮廓点数: {len(contour_points)}
边缘比例: {detection_info['edge_detection']['edge_ratio']:.3f}"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('统计信息')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "05_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. 显示结果
        logger.info("=== 测试结果 ===")
        logger.info(f"原始图像尺寸: {original_image.shape}")
        logger.info(f"ROI区域尺寸: {roi_image.shape}")
        logger.info(f"传统Canny边缘像素: {traditional_edge_count}")
        logger.info(f"改进Canny边缘像素: {advanced_edge_count}")
        logger.info(f"改进算法边缘比例: {detection_info['edge_detection']['edge_ratio']:.3f}")
        logger.info(f"提取轮廓点数: {len(contour_points)}")
        
        logger.info("=== 测试完成 ===")
        logger.info(f"结果已保存到: {output_dir}")
        logger.info("生成的文件:")
        logger.info("  - 01_original.jpg: 原始图像")
        logger.info("  - 02_roi.jpg: ROI区域")
        logger.info("  - 03_advanced_canny.jpg: 改进Canny结果")
        logger.info("  - 04_traditional_canny.jpg: 传统Canny结果")
        logger.info("  - 05_comparison.png: 对比分析图")
        
        return True
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_advanced_canny()
    if success:
        print("✅ 改进Canny算法测试成功!")
    else:
        print("❌ 改进Canny算法测试失败!")