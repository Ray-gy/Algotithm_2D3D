"""
图像预处理模块

负责显微镜图像的预处理，包括去噪、尺寸调整等操作。
"""

import cv2
import numpy as np
from typing import Optional, Union
from loguru import logger


class ImageProcessor:
    """图像预处理器"""
    
    def __init__(self, config: dict):
        """
        初始化图像处理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            加载的图像数组 (BGR格式)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            logger.info(f"成功加载图像: {image_path}, 尺寸: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"加载图像失败: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, 
                    target_width: Optional[int] = None,
                    target_height: Optional[int] = None,
                    keep_aspect: bool = True) -> np.ndarray:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            keep_aspect: 是否保持宽高比
            
        Returns:
            调整尺寸后的图像
        """
        resize_config = self.preprocessing_config.get('resize', {})
        
        if not resize_config.get('enabled', False):
            return image
            
        h, w = image.shape[:2]
        target_w = target_width or resize_config.get('target_width', w)
        target_h = target_height or resize_config.get('target_height', h)
        
        if keep_aspect or resize_config.get('keep_aspect', True):
            # 保持宽高比
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"图像尺寸调整: {image.shape[:2]} -> {resized.shape[:2]}")
        else:
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            logger.info(f"图像尺寸调整: {image.shape[:2]} -> {resized.shape[:2]}")
            
        return resized
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪处理
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        denoise_config = self.preprocessing_config.get('denoise', {})
        method = denoise_config.get('method', 'bilateral')
        
        if method == 'bilateral':
            # 双边滤波 - 保持边缘的同时去噪
            d = denoise_config.get('bilateral_d', 9)
            sigma_color = denoise_config.get('bilateral_sigma_color', 75)
            sigma_space = denoise_config.get('bilateral_sigma_space', 75)
            
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        elif method == 'gaussian':
            # 高斯滤波
            kernel_size = denoise_config.get('gaussian_kernel', 5)
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif method == 'median':
            # 中值滤波
            kernel_size = denoise_config.get('median_kernel', 5)
            denoised = cv2.medianBlur(image, kernel_size)
            
        else:
            logger.warning(f"未知的去噪方法: {method}, 跳过去噪处理")
            return image
            
        logger.info(f"使用{method}方法完成图像去噪")
        return denoised
    
    def preprocess_image(self, image: Union[str, np.ndarray],
                        resize_params: Optional[dict] = None) -> np.ndarray:
        """
        完整的图像预处理流程
        
        Args:
            image: 输入图像路径或图像数组
            resize_params: 尺寸调整参数
            
        Returns:
            预处理后的图像
        """
        # 加载图像（如果输入是路径）
        if isinstance(image, str):
            processed_image = self.load_image(image)
        else:
            processed_image = image.copy()
        
        # 尺寸调整
        if resize_params:
            processed_image = self.resize_image(processed_image, **resize_params)
        else:
            processed_image = self.resize_image(processed_image)
        
        # 去噪处理
        processed_image = self.denoise_image(processed_image)
        
        logger.info("图像预处理完成")
        return processed_image
    
    def convert_color_space(self, image: np.ndarray, 
                           target_space: str = 'hsv') -> np.ndarray:
        """
        颜色空间转换
        
        Args:
            image: 输入图像 (BGR格式)
            target_space: 目标颜色空间 ('hsv', 'rgb', 'gray', 'lab')
            
        Returns:
            转换后的图像
        """
        if target_space.lower() == 'hsv':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_space.lower() == 'rgb':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif target_space.lower() == 'gray':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif target_space.lower() == 'lab':
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            logger.warning(f"不支持的颜色空间: {target_space}")
            return image
            
        logger.debug(f"颜色空间转换: BGR -> {target_space.upper()}")
        return converted