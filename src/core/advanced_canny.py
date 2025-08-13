"""
改进的Canny边缘检测算法

基于自适应高斯滤波器的改进Canny算法，适用于牙齿显微镜图像的边缘检测。
主要改进：
1. 自适应高斯滤波器 - 根据图像局部特征自动调整滤波参数
2. 能量函数优化 - 平衡平滑效果和细节保留
3. 针对牙齿图像特性优化
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class AdaptiveCanny:
    """改进的Canny边缘检测器"""
    
    def __init__(self, config: dict):
        """
        初始化改进的Canny检测器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.contour_config = config.get('contour_extraction', {})
        self.adaptive_config = self.contour_config.get('adaptive_gaussian', {})
        self.canny_config = self.contour_config.get('canny', {})
        
    def adaptive_gaussian_filter(self, image: np.ndarray) -> np.ndarray:
        """
        自适应高斯滤波
        
        根据图像局部特征自动选择最佳的高斯滤波参数σ
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # 获取配置参数
        sigma_min = self.adaptive_config.get('sigma_min', 0.5)
        sigma_max = self.adaptive_config.get('sigma_max', 2.0)
        step_size = self.adaptive_config.get('step_size', 0.1)
        constant_c = self.adaptive_config.get('constant_c', 10.0)
        
        h, w = image.shape
        filtered_image = np.zeros_like(image, dtype=np.float32)
        
        # 为每个像素选择最佳的sigma值
        logger.info("开始自适应高斯滤波处理...")
        
        # 创建sigma值数组
        sigma_values = np.arange(sigma_min, sigma_max + step_size, step_size)
        
        # 预先计算不同sigma值下的高斯滤波结果
        filtered_results = {}
        for sigma in sigma_values:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            filtered_results[sigma] = cv2.GaussianBlur(
                image.astype(np.float32), (kernel_size, kernel_size), sigma
            )
        
        # 为每个像素选择最佳sigma
        best_sigma_map = np.zeros((h, w))
        
        # 滑动窗口大小
        window_size = 9
        half_window = window_size // 2
        
        for i in range(half_window, h - half_window):
            for j in range(half_window, w - half_window):
                # 提取局部窗口
                local_window = image[i-half_window:i+half_window+1, 
                                   j-half_window:j+half_window+1].astype(np.float32)
                
                best_sigma = sigma_min
                min_energy = float('inf')
                
                # 计算不同sigma下的能量函数
                for sigma in sigma_values:
                    # 局部滤波窗口
                    filtered_window = filtered_results[sigma][i-half_window:i+half_window+1,
                                                           j-half_window:j+half_window+1]
                    
                    # 计算残差
                    residual = local_window - filtered_window
                    epsilon_squared = np.mean(residual ** 2)
                    
                    # 能量函数: B = c/σ² + ε²
                    energy = constant_c / (sigma ** 2) + epsilon_squared
                    
                    if energy < min_energy:
                        min_energy = energy
                        best_sigma = sigma
                
                best_sigma_map[i, j] = best_sigma
                filtered_image[i, j] = filtered_results[best_sigma][i, j]
        
        # 处理边界像素（使用最近邻的sigma值）
        for i in range(h):
            for j in range(w):
                if i < half_window or i >= h - half_window or j < half_window or j >= w - half_window:
                    # 找到最近的内部像素的sigma值
                    ii = max(half_window, min(i, h - half_window - 1))
                    jj = max(half_window, min(j, w - half_window - 1))
                    sigma = best_sigma_map[ii, jj]
                    filtered_image[i, j] = filtered_results[sigma][i, j]
        
        logger.info("自适应高斯滤波完成")
        return filtered_image.astype(np.uint8)
    
    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算图像梯度幅值和方向
        
        Args:
            image: 输入图像
            
        Returns:
            梯度幅值、X方向梯度、Y方向梯度
        """
        # 使用Sobel算子计算梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return magnitude, grad_x, grad_y
    
    def non_maximum_suppression(self, magnitude: np.ndarray, 
                               grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """
        非极大值抑制
        
        Args:
            magnitude: 梯度幅值
            grad_x: X方向梯度
            grad_y: Y方向梯度
            
        Returns:
            抑制后的梯度幅值
        """
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # 计算梯度方向（角度）
        angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi
        angle[angle < 0] += 180  # 转换到[0, 180)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                try:
                    q = 255
                    r = 255
                    
                    # 角度0-22.5和157.5-180
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = magnitude[i, j + 1]
                        r = magnitude[i, j - 1]
                    # 角度22.5-67.5
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = magnitude[i + 1, j - 1]
                        r = magnitude[i - 1, j + 1]
                    # 角度67.5-112.5
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = magnitude[i + 1, j]
                        r = magnitude[i - 1, j]
                    # 角度112.5-157.5
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = magnitude[i - 1, j - 1]
                        r = magnitude[i + 1, j + 1]
                    
                    if magnitude[i, j] >= q and magnitude[i, j] >= r:
                        suppressed[i, j] = magnitude[i, j]
                    else:
                        suppressed[i, j] = 0
                        
                except IndexError:
                    pass
        
        return suppressed
    
    def double_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        双阈值处理
        
        Args:
            image: 经过非极大值抑制的图像
            
        Returns:
            二值化边缘图像
        """
        if self.canny_config.get('auto_threshold', True):
            # 自动计算阈值
            high_threshold = np.max(image) * 0.2
            low_threshold = high_threshold * 0.5
        else:
            high_threshold = self.canny_config.get('threshold_high', 150)
            low_threshold = self.canny_config.get('threshold_low', 50)
        
        logger.debug(f"双阈值: 低阈值={low_threshold:.2f}, 高阈值={high_threshold:.2f}")
        
        # 创建输出图像
        result = np.zeros_like(image)
        
        # 强边缘
        strong_edges = (image >= high_threshold)
        result[strong_edges] = 255
        
        # 弱边缘
        weak_edges = ((image >= low_threshold) & (image < high_threshold))
        result[weak_edges] = 75  # 临时值，用于后续处理
        
        # 边缘跟踪
        result = self._edge_tracking(result, weak_edges, strong_edges)
        
        return result
    
    def _edge_tracking(self, image: np.ndarray, weak_edges: np.ndarray, 
                      strong_edges: np.ndarray) -> np.ndarray:
        """
        边缘跟踪，连接弱边缘到强边缘
        
        Args:
            image: 当前图像
            weak_edges: 弱边缘掩码
            strong_edges: 强边缘掩码
            
        Returns:
            最终的边缘图像
        """
        h, w = image.shape
        
        # 8邻域
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                     (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # 使用队列进行边缘跟踪
        from collections import deque
        queue = deque()
        
        # 将所有强边缘加入队列
        strong_coords = np.where(strong_edges)
        for i, j in zip(strong_coords[0], strong_coords[1]):
            queue.append((i, j))
        
        # BFS边缘跟踪
        while queue:
            y, x = queue.popleft()
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < h and 0 <= nx < w and 
                    weak_edges[ny, nx] and image[ny, nx] == 75):
                    image[ny, nx] = 255  # 连接到强边缘
                    queue.append((ny, nx))
        
        # 移除未连接的弱边缘
        image[image == 75] = 0
        
        return image.astype(np.uint8)
    
    def detect_edges(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        完整的改进Canny边缘检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            边缘图像和处理信息
        """
        logger.info("开始改进Canny边缘检测")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. 自适应高斯滤波
        logger.info("步骤1: 自适应高斯滤波")
        filtered = self.adaptive_gaussian_filter(gray)
        
        # 2. 计算梯度
        logger.info("步骤2: 计算梯度")
        magnitude, grad_x, grad_y = self.compute_gradients(filtered)
        
        # 3. 非极大值抑制
        logger.info("步骤3: 非极大值抑制")
        suppressed = self.non_maximum_suppression(magnitude, grad_x, grad_y)
        
        # 4. 双阈值处理和边缘跟踪
        logger.info("步骤4: 双阈值处理和边缘跟踪")
        edges = self.double_threshold(suppressed)
        
        # 统计信息
        info = {
            'total_pixels': edges.size,
            'edge_pixels': np.count_nonzero(edges),
            'edge_ratio': np.count_nonzero(edges) / edges.size
        }
        
        logger.info(f"边缘检测完成: 边缘像素={info['edge_pixels']}, "
                   f"边缘比例={info['edge_ratio']:.3f}")
        
        return edges, info
    
    def extract_contour_points(self, edge_image: np.ndarray) -> np.ndarray:
        """
        从边缘图像中提取轮廓点集
        
        使用改进的方法：先进行形态学操作连接断裂的边缘，再提取轮廓
        
        Args:
            edge_image: 边缘二值图像
            
        Returns:
            轮廓点集 (N, 2)
        """
        # 确保图像是8位无符号整数类型
        if edge_image.dtype != np.uint8:
            edge_image = edge_image.astype(np.uint8)
        
        # 1. 先进行形态学闭运算，连接断裂的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_edges = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 2. 膨胀操作，增强边缘连通性
        dilated_edges = cv2.dilate(closed_edges, kernel, iterations=1)
        
        # 3. 查找轮廓
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("未找到任何轮廓")
            # 如果形态学处理后仍然没有轮廓，直接从边缘像素点提取
            return self._extract_points_from_edges(edge_image)
        
        # 4. 过滤轮廓（放宽条件）
        filter_config = self.contour_config.get('contour_filter', {})
        min_area = max(filter_config.get('min_area', 50), 10)  # 最小面积降低到10
        max_area = filter_config.get('max_area', 500000)
        min_perimeter = max(filter_config.get('min_perimeter', 20), 5)  # 最小周长降低到5
        
        filtered_contours = []
        logger.debug(f"轮廓过滤条件: 面积[{min_area}, {max_area}], 周长>={min_perimeter}")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if len(contours) <= 10:  # 只在轮廓少时显示详细信息
                logger.debug(f"轮廓{i}: 面积={area:.2f}, 周长={perimeter:.2f}")
            
            if area >= min_area and perimeter >= min_perimeter and area <= max_area:
                filtered_contours.append(contour)
        
        logger.info(f"找到{len(contours)}个轮廓，过滤后剩余{len(filtered_contours)}个")
        
        if not filtered_contours:
            logger.warning("过滤后没有有效轮廓，尝试提取边缘像素点")
            return self._extract_points_from_edges(edge_image)
        
        # 5. 选择最大的轮廓
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # 6. 轮廓逼近，减少点数
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 转换为点集格式 (N, 2)
        points = approx_contour.reshape(-1, 2)
        
        logger.info(f"提取轮廓点集: 最大轮廓面积={largest_area:.2f}, 逼近后{len(points)}个点")
        
        return points
    
    def _extract_points_from_edges(self, edge_image: np.ndarray) -> np.ndarray:
        """
        当无法找到有效轮廓时，直接从边缘像素提取点集
        
        Args:
            edge_image: 边缘二值图像
            
        Returns:
            边缘点集 (N, 2)
        """
        # 找到所有边缘像素点
        edge_points = np.where(edge_image > 0)
        
        if len(edge_points[0]) == 0:
            logger.warning("没有找到任何边缘像素")
            return np.array([])
        
        # 转换为(x, y)格式
        points = np.column_stack((edge_points[1], edge_points[0]))
        
        # 对点进行采样，减少密度
        if len(points) > 500:
            # 随机采样500个点
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
        
        logger.info(f"从边缘像素提取点集: {len(points)}个点")
        
        return points