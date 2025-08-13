# 改进Canny算法使用说明

## 概述

改进的Canny算法基于论文《基于内窥镜与CT图像融合的机器人辅助根管治疗术中导航研究》，主要特点是使用自适应高斯滤波器来替代传统的固定尺寸高斯滤波器，能够更好地处理牙齿显微镜图像的边缘检测。

## 主要改进

### 1. 自适应高斯滤波
- **传统方法**: 使用固定的σ值进行高斯滤波
- **改进方法**: 根据图像局部特征自动选择最佳的σ值
- **优势**: 在去噪的同时更好地保留边缘细节

### 2. 能量函数优化
使用能量函数来平衡平滑效果和细节保留：

```
B(σ) = c/σ² + ε²
```

其中：
- `c`: 常数项，控制平滑程度
- `σ`: 高斯滤波标准差
- `ε`: 滤波前后的残差

### 3. 针对牙齿图像优化
- 适应牙齿表面的光照变化
- 处理绿色橡皮障背景的干扰
- 优化轮廓连续性

## 配置参数

### adaptive_gaussian 参数
```yaml
adaptive_gaussian:
  sigma_min: 0.5          # 最小标准差
  sigma_max: 2.0          # 最大标准差  
  step_size: 0.1          # 搜索步长
  constant_c: 10.0        # 能量函数常数项
```

**参数调优建议**:
- `sigma_min/max`: 根据图像噪声程度调整，噪声大时增大范围
- `step_size`: 较小的步长提供更精细的控制，但增加计算时间
- `constant_c`: 较大值倾向于更多平滑，较小值保留更多细节

### canny 参数
```yaml
canny:
  auto_threshold: true    # 自动阈值计算
  threshold_low: 50       # 低阈值（手动模式）
  threshold_high: 150     # 高阈值（手动模式）
  aperture_size: 3        # Sobel核大小
```

### contour_filter 参数
```yaml
contour_filter:
  min_area: 100           # 最小轮廓面积
  max_area: 50000         # 最大轮廓面积
  min_perimeter: 50       # 最小周长
```

## 使用方法

### 基本使用

```python
from core.segmentation import ToothSegmentation
from utils.file_io import load_config

# 加载配置
config = load_config("config/image_processing.yaml")

# 初始化分割器
segmentation = ToothSegmentation(config)

# 处理图像
segmented_image, contour_points, info = segmentation.segment_and_extract_contours(
    image, use_advanced_canny=True
)
```

### 仅边缘检测

```python
# 仅使用改进的Canny算法进行边缘检测
contour_points, detection_info = segmentation.extract_contours_advanced_canny(roi_image)

# 获取边缘图像
edges = detection_info['edges']

# 获取统计信息
edge_count = detection_info['edge_detection']['edge_pixels']
edge_ratio = detection_info['edge_detection']['edge_ratio']
```

## 测试和验证

### 运行测试脚本

```bash
python test_advanced_canny.py
```

测试脚本将：
1. 加载配置和测试图像
2. 执行完整的处理流程
3. 对比传统Canny和改进Canny的结果
4. 生成可视化对比图
5. 输出详细的统计信息

### 运行完整演示

```bash
python scripts/demo.py
```

演示将生成以下输出文件：
- `06_edges.jpg`: 改进Canny边缘检测结果
- `07_contour_points.jpg`: 轮廓点集可视化

## 性能优化建议

### 1. 参数调优
- **图像质量好**: 减小`sigma_max`，提高处理速度
- **噪声较多**: 增大`sigma_max`，提高去噪效果
- **需要精细边缘**: 减小`step_size`，增加`constant_c`
- **需要快速处理**: 增大`step_size`，减少计算量

### 2. 计算优化
- 对于实时应用，可以预先计算不同σ值的滤波结果
- 使用并行处理来加速自适应滤波过程
- 对于批量处理，可以复用滤波结果

### 3. 内存优化
- 处理大图像时，可以分块处理
- 及时释放不需要的中间结果

## 典型问题和解决方案

### 1. 边缘检测不够敏感
- 增加`sigma_max`范围
- 减小`constant_c`值
- 启用自动阈值`auto_threshold: true`

### 2. 检测到太多噪声边缘
- 减小`sigma_max`范围
- 增加`constant_c`值
- 调整轮廓过滤参数

### 3. 轮廓不连续
- 减小双阈值的差距
- 增强边缘跟踪效果
- 优化形态学操作参数

### 4. 处理速度慢
- 增大`step_size`
- 减小σ搜索范围
- 降低图像分辨率进行预处理

## 输出格式

### contour_points
- 格式: `np.ndarray`, shape为`(N, 2)`
- 内容: 轮廓点的(x, y)坐标
- 坐标系: 图像像素坐标系

### detection_info
```python
{
    'edge_detection': {
        'total_pixels': int,      # 总像素数
        'edge_pixels': int,       # 边缘像素数
        'edge_ratio': float       # 边缘比例
    },
    'contour_points_count': int,  # 轮廓点数量
    'edges': np.ndarray          # 边缘二值图像
}
```

## 下一步开发

完成改进Canny算法后，接下来的开发步骤：

1. **3D模型投影**: 实现基于口腔空间约束的3D模型投影
2. **CPD点集配准**: 实现改进的相干点漂移算法
3. **图像融合**: 将2D轮廓与3D模型进行配准融合
4. **实时处理**: 优化算法性能，支持实时处理

## 参考文献

张嘉伟. 基于内窥镜与CT图像融合的机器人辅助根管治疗术中导航研究[D]. 哈尔滨理工大学, 2024.