"""
数学计算工具模块

提供轮廓分析、几何计算等数学工具函数。
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from loguru import logger


def calculate_contour_properties(contour: np.ndarray) -> Dict[str, Any]:
    """
    计算轮廓的几何属性
    
    Args:
        contour: 轮廓点集
        
    Returns:
        轮廓属性字典
    """
    properties = {}
    
    try:
        # 基础属性
        properties['area'] = cv2.contourArea(contour)
        properties['perimeter'] = cv2.arcLength(contour, True)
        
        # 边界框
        x, y, w, h = cv2.boundingRect(contour)
        properties['bounding_box'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        properties['aspect_ratio'] = w / h if h > 0 else 0
        
        # 最小外接矩形
        min_rect = cv2.minAreaRect(contour)
        properties['min_area_rect'] = {
            'center': (float(min_rect[0][0]), float(min_rect[0][1])),
            'size': (float(min_rect[1][0]), float(min_rect[1][1])),
            'angle': float(min_rect[2])
        }
        
        # 重心/质心
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            properties['centroid'] = (cx, cy)
        else:
            properties['centroid'] = None
        
        # 轮廓的凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        properties['hull_area'] = hull_area
        properties['solidity'] = properties['area'] / hull_area if hull_area > 0 else 0
        
        # 等价直径（与轮廓面积相等的圆的直径）
        properties['equivalent_diameter'] = np.sqrt(4 * properties['area'] / np.pi)
        
        # 紧密度（4π*面积/周长²）
        if properties['perimeter'] > 0:
            properties['compactness'] = 4 * np.pi * properties['area'] / (properties['perimeter'] ** 2)
        else:
            properties['compactness'] = 0
        
        # 拟合椭圆（需要至少5个点）
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            properties['fitted_ellipse'] = {
                'center': (float(ellipse[0][0]), float(ellipse[0][1])),
                'axes': (float(ellipse[1][0]), float(ellipse[1][1])),
                'angle': float(ellipse[2])
            }
        else:
            properties['fitted_ellipse'] = None
            
        logger.debug(f"轮廓属性计算完成，面积: {properties['area']:.2f}")
        
    except Exception as e:
        logger.error(f"轮廓属性计算失败: {e}")
        properties = {'error': str(e)}
    
    return properties


def calculate_rmse(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算两个点集之间的均方根误差(RMSE)
    
    Args:
        points1: 第一个点集 (N, 2)
        points2: 第二个点集 (M, 2)
        
    Returns:
        RMSE值
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # 确保点集格式正确
    points1 = points1.reshape(-1, 2).astype(np.float32)
    points2 = points2.reshape(-1, 2).astype(np.float32)
    
    # 对于不等长的点集，使用最近点匹配
    if len(points1) != len(points2):
        # 使用KD树找最近点（简化版本）
        from scipy.spatial.distance import cdist
        
        # 计算距离矩阵
        distances = cdist(points1, points2)
        
        # 为每个点找到最近的匹配点
        min_distances = np.min(distances, axis=1)
        rmse = np.sqrt(np.mean(min_distances ** 2))
    else:
        # 等长点集直接计算
        diff = points1 - points2
        rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return float(rmse)


def normalize_contour(contour: np.ndarray, 
                     target_points: int = 100) -> np.ndarray:
    """
    标准化轮廓点数
    
    Args:
        contour: 输入轮廓
        target_points: 目标点数
        
    Returns:
        标准化后的轮廓
    """
    # 计算轮廓周长
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return contour
    
    # 重新采样轮廓点
    epsilon = perimeter / target_points
    normalized_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # 如果点数不够，进行插值
    if len(normalized_contour) < target_points:
        # 线性插值增加点数
        normalized_contour = interpolate_contour(normalized_contour, target_points)
    
    logger.debug(f"轮廓标准化: {len(contour)} -> {len(normalized_contour)} 点")
    return normalized_contour


def interpolate_contour(contour: np.ndarray, target_points: int) -> np.ndarray:
    """
    对轮廓进行插值以达到目标点数
    
    Args:
        contour: 输入轮廓
        target_points: 目标点数
        
    Returns:
        插值后的轮廓
    """
    contour = contour.squeeze()
    
    if len(contour) >= target_points:
        # 如果点数已足够，进行降采样
        indices = np.linspace(0, len(contour) - 1, target_points, dtype=int)
        return contour[indices].reshape(-1, 1, 2)
    
    # 计算累积长度
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(distances)])
    
    # 创建等间距的采样点
    total_length = cumulative_length[-1]
    sample_lengths = np.linspace(0, total_length, target_points)
    
    # 插值
    interpolated_x = np.interp(sample_lengths, cumulative_length, contour[:, 0])
    interpolated_y = np.interp(sample_lengths, cumulative_length, contour[:, 1])
    
    interpolated_contour = np.column_stack([interpolated_x, interpolated_y])
    return interpolated_contour.reshape(-1, 1, 2).astype(np.int32)


def calculate_contour_similarity(contour1: np.ndarray, 
                               contour2: np.ndarray,
                               method: str = 'hausdorff') -> float:
    """
    计算两个轮廓的相似度
    
    Args:
        contour1: 第一个轮廓
        contour2: 第二个轮廓  
        method: 相似度计算方法 ('hausdorff', 'shape_match', 'rmse')
        
    Returns:
        相似度值（值越小越相似）
    """
    try:
        if method == 'hausdorff':
            # Hausdorff距离
            return cv2.createHausdorffDistanceExtractor().computeDistance(contour1, contour2)
        
        elif method == 'shape_match':
            # 形状匹配
            return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
        
        elif method == 'rmse':
            # RMSE相似度
            points1 = contour1.reshape(-1, 2)
            points2 = contour2.reshape(-1, 2)
            return calculate_rmse(points1, points2)
        
        else:
            logger.warning(f"未知的相似度计算方法: {method}")
            return float('inf')
            
    except Exception as e:
        logger.error(f"轮廓相似度计算失败: {e}")
        return float('inf')


def get_contour_signature(contour: np.ndarray, 
                         signature_type: str = 'distance') -> np.ndarray:
    """
    计算轮廓特征签名
    
    Args:
        contour: 输入轮廓
        signature_type: 签名类型 ('distance', 'curvature', 'angle')
        
    Returns:
        轮廓特征签名数组
    """
    contour = contour.squeeze()
    
    if signature_type == 'distance':
        # 距离签名：每个点到质心的距离
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            centroid = np.array([cx, cy])
            
            distances = np.linalg.norm(contour - centroid, axis=1)
            return distances
        else:
            return np.zeros(len(contour))
    
    elif signature_type == 'curvature':
        # 曲率签名
        # 简化的曲率计算
        n = len(contour)
        curvatures = []
        
        for i in range(n):
            p1 = contour[(i-1) % n]
            p2 = contour[i]
            p3 = contour[(i+1) % n]
            
            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算角度变化（简化的曲率）
            cross = np.cross(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                curvature = cross / (norm1 * norm2)
            else:
                curvature = 0
                
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    elif signature_type == 'angle':
        # 角度签名
        n = len(contour)
        angles = []
        
        for i in range(n):
            p1 = contour[(i-1) % n]
            p2 = contour[i]
            p3 = contour[(i+1) % n]
            
            # 计算角度
            v1 = p1 - p2
            v2 = p3 - p2
            
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            angles.append(angle)
        
        return np.array(angles)
    
    else:
        logger.warning(f"未知的签名类型: {signature_type}")
        return np.array([])


def filter_contours_by_properties(contours: list, 
                                min_area: float = 100,
                                max_area: float = 50000,
                                min_perimeter: float = 50) -> list:
    """
    根据几何属性过滤轮廓
    
    Args:
        contours: 轮廓列表
        min_area: 最小面积
        max_area: 最大面积  
        min_perimeter: 最小周长
        
    Returns:
        过滤后的轮廓列表
    """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if min_area <= area <= max_area and perimeter >= min_perimeter:
            filtered_contours.append(contour)
    
    logger.info(f"轮廓过滤: {len(contours)} -> {len(filtered_contours)}")
    return filtered_contours