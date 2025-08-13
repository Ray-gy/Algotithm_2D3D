"""
图像分割模块

专门用于显微镜牙齿图像的分割，利用绿色橡皮障背景进行快速分割。
包含改进的Canny边缘检测算法。
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from loguru import logger
from .advanced_canny import AdaptiveCanny


class ToothSegmentation:
    """牙齿分割器 - 针对显微镜图像优化"""
    
    def __init__(self, config: dict):
        """
        初始化分割器
        
        Args:
            config: 分割配置参数
        """
        self.config = config
        self.segmentation_config = config.get('segmentation', {})
        
        # 初始化改进的Canny边缘检测器
        self.adaptive_canny = AdaptiveCanny(config)
        
    def create_color_mask(self, image: np.ndarray) -> np.ndarray:
        """
        基于颜色创建掩码（针对绿色橡皮障背景）
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            二值化掩码 (0: 背景, 255: 前景牙齿)
        """
        color_config = self.segmentation_config.get('color_based', {})
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 获取HSV阈值范围
        hsv_lower = np.array(color_config.get('hsv_lower', [40, 40, 40]))
        hsv_upper = np.array(color_config.get('hsv_upper', [80, 255, 255]))
        
        # 创建颜色掩码（绿色区域）
        green_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # 根据配置决定是否反转掩码
        if color_config.get('invert_mask', True):
            # 反转掩码：保留非绿色区域（牙齿区域）
            mask = cv2.bitwise_not(green_mask)
            logger.debug("创建反转颜色掩码（保留非绿色区域）")
        else:
            mask = green_mask
            logger.debug("创建颜色掩码（绿色区域）")
            
        return mask
    
    def apply_morphology_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学操作优化掩码
        
        Args:
            mask: 输入二值化掩码
            
        Returns:
            优化后的掩码
        """
        morph_config = self.segmentation_config.get('morphology', {})
        
        # 创建结构元素
        kernel_size = morph_config.get('kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (kernel_size, kernel_size))
        
        # 开运算：去除小的噪声点
        opening_iter = morph_config.get('opening_iterations', 2)
        if opening_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                                  iterations=opening_iter)
            logger.debug(f"应用开运算，迭代次数: {opening_iter}")
        
        # 闭运算：填充小的空洞
        closing_iter = morph_config.get('closing_iterations', 3)
        if closing_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=closing_iter)
            logger.debug(f"应用闭运算，迭代次数: {closing_iter}")
        
        return mask
    
    def find_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        找到最大的连通区域（假设为主要的牙齿区域）
        
        Args:
            mask: 二值化掩码
            
        Returns:
            最大轮廓的点集，如果没有找到则返回None
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("未找到任何轮廓")
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        logger.info(f"找到最大轮廓，面积: {area:.2f} 像素²")
        return largest_contour
    
    def create_refined_mask(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        基于最大轮廓创建精细化掩码
        
        Args:
            image: 原始图像
            contour: 轮廓点集
            
        Returns:
            精细化的二值掩码
        """
        # 创建空白掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 填充轮廓区域
        cv2.fillPoly(mask, [contour], 255)
        
        logger.debug("创建基于轮廓的精细化掩码")
        return mask
    
    def segment_tooth(self, image: np.ndarray, 
                     return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        完整的牙齿分割流程
        
        Args:
            image: 输入图像 (BGR格式)
            return_mask: 是否返回掩码
            
        Returns:
            分割后的牙齿图像，以及可选的掩码
        """
        logger.info("开始牙齿分割处理")
        
        # 1. 创建基于颜色的初始掩码
        initial_mask = self.create_color_mask(image)
        
        # 2. 应用形态学操作优化掩码
        refined_mask = self.apply_morphology_operations(initial_mask)
        
        # 3. 找到最大连通区域
        largest_contour = self.find_largest_contour(refined_mask)
        
        if largest_contour is not None:
            # 4. 基于最大轮廓创建最终掩码
            final_mask = self.create_refined_mask(image, largest_contour)
        else:
            logger.warning("未找到有效轮廓，使用原始掩码")
            final_mask = refined_mask
        
        # 5. 应用掩码到原图像
        segmented_image = cv2.bitwise_and(image, image, mask=final_mask)
        
        logger.info("牙齿分割处理完成")
        
        if return_mask:
            return segmented_image, final_mask
        else:
            return segmented_image, None
    
    def extract_roi_bbox(self, mask: np.ndarray, 
                        padding: int = 10) -> Tuple[int, int, int, int]:
        """
        从掩码中提取感兴趣区域的边界框
        
        Args:
            mask: 二值化掩码
            padding: 边界框扩展像素数
            
        Returns:
            边界框坐标 (x, y, width, height)
        """
        # 找到非零像素的位置
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            logger.warning("掩码中没有前景像素")
            return 0, 0, mask.shape[1], mask.shape[0]
        
        # 计算边界框
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 添加填充并确保不超出图像边界
        h, w = mask.shape
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        width = x_max - x_min
        height = y_max - y_min
        
        logger.debug(f"提取ROI边界框: ({x_min}, {y_min}, {width}, {height})")
        return x_min, y_min, width, height
    
    def crop_to_roi(self, image: np.ndarray, mask: np.ndarray,
                   padding: int = 10) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        将图像裁剪到感兴趣区域
        
        Args:
            image: 原始图像
            mask: 对应的掩码
            padding: 边界框扩展像素数
            
        Returns:
            裁剪后的图像、掩码和边界框信息
        """
        x, y, w, h = self.extract_roi_bbox(mask, padding)
        
        # 裁剪图像和掩码
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        logger.info(f"图像裁剪到ROI: 原尺寸{image.shape[:2]} -> 新尺寸{cropped_image.shape[:2]}")
        
        return cropped_image, cropped_mask, (x, y, w, h)
    
    def extract_contours_advanced_canny(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        使用改进的Canny算法进行边缘检测和轮廓提取
        
        Args:
            image: 输入图像（已分割的牙齿图像）
            
        Returns:
            轮廓点集和检测信息
        """
        logger.info("开始使用改进Canny算法进行轮廓提取")
        
        # 使用改进的Canny算法检测边缘
        edges, edge_info = self.adaptive_canny.detect_edges(image)
        
        # 提取轮廓点集
        contour_points = self.adaptive_canny.extract_contour_points(edges)
        
        # 组合检测信息
        detection_info = {
            'edge_detection': edge_info,
            'contour_points_count': len(contour_points) if len(contour_points) > 0 else 0,
            'edges': edges  # 包含边缘图像用于可视化
        }
        
        logger.info(f"轮廓提取完成: 提取到{detection_info['contour_points_count']}个轮廓点")
        
        return contour_points, detection_info
    
    def segment_and_extract_contours(self, image: np.ndarray, 
                                   use_advanced_canny: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        完整的牙齿分割和轮廓提取流程
        
        Args:
            image: 输入图像 (BGR格式)
            use_advanced_canny: 是否使用改进的Canny算法
            
        Returns:
            分割后的牙齿图像、轮廓点集和处理信息
        """
        logger.info("开始完整的牙齿分割和轮廓提取流程")
        
        # 1. 牙齿分割
        segmented_image, mask = self.segment_tooth(image, return_mask=True)
        
        # 2. ROI裁剪
        roi_image, roi_mask, bbox = self.crop_to_roi(segmented_image, mask)
        
        # 3. 轮廓提取
        if use_advanced_canny:
            contour_points, contour_info = self.extract_contours_advanced_canny(roi_image)
        else:
            # 使用传统方法（可以后续实现）
            logger.warning("传统轮廓提取方法暂未实现，使用改进Canny算法")
            contour_points, contour_info = self.extract_contours_advanced_canny(roi_image)
        
        # 将轮廓点坐标转换回原图坐标系
        if len(contour_points) > 0:
            contour_points[:, 0] += bbox[0]  # 加上x偏移
            contour_points[:, 1] += bbox[1]  # 加上y偏移
        
        # 组合处理信息
        processing_info = {
            'segmentation': {
                'roi_bbox': bbox,
                'roi_size': roi_image.shape[:2],
                'mask_area': np.count_nonzero(mask)
            },
            'contour_extraction': contour_info
        }
        
        logger.info("完整处理流程完成")
        
        return segmented_image, contour_points, processing_info