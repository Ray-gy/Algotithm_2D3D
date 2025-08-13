"""
文件I/O工具模块

处理配置文件加载、结果保存等文件操作。
"""

import yaml
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any, List, Union
from loguru import logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径 (.yaml 或 .json)
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def save_results(data: Dict[str, Any], output_path: Union[str, Path], 
                format_type: str = 'json') -> None:
    """
    保存处理结果
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
        format_type: 输出格式 ('json', 'pickle', 'yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format_type.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        elif format_type.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
                
        elif format_type.lower() in ['yaml', 'yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, default_flow_style=False, 
                              allow_unicode=True)
        else:
            raise ValueError(f"不支持的输出格式: {format_type}")
        
        logger.info(f"结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        raise


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        Path对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"确保目录存在: {dir_path}")
    return dir_path


def get_image_files(directory: Union[str, Path], 
                   extensions: List[str] = None) -> List[Path]:
    """
    获取目录下所有图像文件
    
    Args:
        directory: 图像目录
        extensions: 支持的文件扩展名列表
        
    Returns:
        图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    image_files.sort()
    logger.info(f"在 {directory} 中找到 {len(image_files)} 个图像文件")
    
    return image_files


def load_image_list(list_file: Union[str, Path]) -> List[str]:
    """
    从文件加载图像路径列表
    
    Args:
        list_file: 包含图像路径的文本文件
        
    Returns:
        图像路径列表
    """
    list_file = Path(list_file)
    
    if not list_file.exists():
        raise FileNotFoundError(f"图像列表文件不存在: {list_file}")
    
    try:
        with open(list_file, 'r', encoding='utf-8') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"从 {list_file} 加载了 {len(image_paths)} 个图像路径")
        return image_paths
        
    except Exception as e:
        logger.error(f"加载图像列表失败: {e}")
        raise


def create_output_structure(base_path: Union[str, Path], 
                          subdirs: List[str] = None) -> Dict[str, Path]:
    """
    创建输出目录结构
    
    Args:
        base_path: 基础输出路径
        subdirs: 子目录名称列表
        
    Returns:
        目录路径字典
    """
    if subdirs is None:
        subdirs = ['images', 'logs', 'metrics', 'cache']
    
    base_path = Path(base_path)
    paths = {'base': ensure_directory(base_path)}
    
    for subdir in subdirs:
        paths[subdir] = ensure_directory(base_path / subdir)
    
    logger.info(f"创建输出目录结构: {base_path}")
    return paths