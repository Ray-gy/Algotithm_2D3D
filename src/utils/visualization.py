"""
可视化工具模块

提供图像处理结果的可视化功能。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from loguru import logger


class ImageVisualizer:
    """图像可视化工具类"""
    
    def __init__(self, config: dict):
        """
        初始化可视化器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.vis_config = config.get('visualization', {})
        self.line_thickness = self.vis_config.get('line_thickness', 2)
        self.colors = self.vis_config.get('colors', {
            'contour': [0, 255, 0],
            'mask': [255, 0, 0], 
            'background': [0, 0, 255]
        })
        
    def create_side_by_side(self, *images: np.ndarray, 
                           titles: Optional[List[str]] = None,
                           padding: int = 10) -> np.ndarray:
        """
        创建并排显示的图像对比
        
        Args:
            *images: 要对比的图像
            titles: 图像标题列表
            padding: 图像间的间距
            
        Returns:
            合并后的对比图像
        """
        if len(images) == 0:
            raise ValueError("至少需要一张图像")
        
        # 确保所有图像都是3通道
        processed_images = []
        for img in images:
            if len(img.shape) == 2:  # 灰度图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            processed_images.append(img)
        
        # 统一图像高度（以最小高度为准）
        min_height = min(img.shape[0] for img in processed_images)
        resized_images = []
        
        for img in processed_images:
            if img.shape[0] != min_height:
                scale = min_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, min_height))
            resized_images.append(img)
        
        # 计算总宽度
        total_width = sum(img.shape[1] for img in resized_images) + padding * (len(resized_images) - 1)
        
        # 创建合并画布
        combined = np.zeros((min_height, total_width, 3), dtype=np.uint8)
        
        # 拼接图像
        x_offset = 0
        for i, img in enumerate(resized_images):
            combined[:, x_offset:x_offset + img.shape[1]] = img
            
            # 添加标题
            if titles and i < len(titles):
                cv2.putText(combined, titles[i], (x_offset + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            x_offset += img.shape[1] + padding
        
        return combined
    
    def create_processing_comparison(self, original: np.ndarray,
                                   preprocessed: np.ndarray,
                                   segmented: np.ndarray,
                                   roi: np.ndarray) -> np.ndarray:
        """
        创建处理流程对比图
        
        Args:
            original: 原始图像
            preprocessed: 预处理图像
            segmented: 分割图像
            roi: ROI图像
            
        Returns:
            处理流程对比图
        """
        titles = ['Original', 'Preprocessed', 'Segmented', 'ROI']
        return self.create_side_by_side(original, preprocessed, segmented, roi, titles=titles)
    
    def draw_contour_on_image(self, image: np.ndarray, 
                             contour: np.ndarray,
                             color: Optional[Tuple[int, int, int]] = None,
                             thickness: Optional[int] = None) -> np.ndarray:
        """
        在图像上绘制轮廓
        
        Args:
            image: 输入图像
            contour: 轮廓点集
            color: 轮廓颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            绘制轮廓后的图像
        """
        result_image = image.copy()
        
        if color is None:
            color = tuple(self.colors.get('contour', [0, 255, 0]))
        
        if thickness is None:
            thickness = self.line_thickness
            
        cv2.drawContours(result_image, [contour], -1, color, thickness)
        
        return result_image
    
    def create_mask_overlay(self, image: np.ndarray, mask: np.ndarray,
                           alpha: float = 0.5,
                           mask_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        创建掩码叠加效果
        
        Args:
            image: 原始图像
            mask: 二值掩码
            alpha: 透明度
            mask_color: 掩码颜色
            
        Returns:
            叠加掩码后的图像
        """
        if mask_color is None:
            mask_color = tuple(self.colors.get('mask', [255, 0, 0]))
        
        # 确保输入图像是3通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = mask_color
        
        # 叠加
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def plot_processing_stages(self, stages_data: dict, save_path: Optional[str] = None):
        """
        使用matplotlib绘制处理阶段结果
        
        Args:
            stages_data: 各阶段数据字典 {'stage_name': image, ...}
            save_path: 保存路径
        """
        n_stages = len(stages_data)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (stage_name, image) in enumerate(stages_data.items()):
            if i >= 4:  # 最多显示4个阶段
                break
                
            # 转换颜色空间用于显示
            if len(image.shape) == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image
                
            axes[i].imshow(display_image, cmap='gray' if len(image.shape) == 2 else None)
            axes[i].set_title(stage_name)
            axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(n_stages, 4):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"处理阶段图表已保存: {save_path}")
        
        plt.show()
    
    def create_grid_view(self, images: List[np.ndarray], 
                        titles: Optional[List[str]] = None,
                        grid_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        创建网格视图
        
        Args:
            images: 图像列表
            titles: 标题列表
            grid_shape: 网格形状 (rows, cols)
            
        Returns:
            网格合并后的图像
        """
        n_images = len(images)
        
        if grid_shape is None:
            # 自动计算网格形状
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            grid_shape = (rows, cols)
        
        rows, cols = grid_shape
        
        # 确保所有图像尺寸一致
        if images:
            target_height = images[0].shape[0]
            target_width = images[0].shape[1]
            
            resized_images = []
            for img in images:
                if img.shape[:2] != (target_height, target_width):
                    img = cv2.resize(img, (target_width, target_height))
                
                # 确保3通道
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                resized_images.append(img)
            
            # 创建网格
            grid_height = target_height * rows
            grid_width = target_width * cols
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, img in enumerate(resized_images):
                row = i // cols
                col = i % cols
                
                y1 = row * target_height
                y2 = y1 + target_height
                x1 = col * target_width  
                x2 = x1 + target_width
                
                grid_image[y1:y2, x1:x2] = img
                
                # 添加标题
                if titles and i < len(titles):
                    cv2.putText(grid_image, titles[i], (x1 + 10, y1 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return grid_image
        
        return np.zeros((100, 100, 3), dtype=np.uint8)