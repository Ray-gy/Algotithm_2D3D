# 牙齿实时二维-三维图像配准算法

## 项目简介
基于显微镜图像的牙齿实时二维-三维配准算法，用于根管治疗的术中导航。本项目复现并改进了论文《基于内窥镜与CT图像融合的机器人辅助根管治疗术中导航研究》中的关键算法。

## 技术特点
- **显微镜图像优化**: 针对显微镜拍摄的高质量牙齿图像进行特殊优化
- **改进的分割算法**: 利用绿色橡皮障背景实现快速精确分割
- **自适应轮廓提取**: 基于论文的改进Canny算法实现轮廓提取
- **实时配准**: 支持实时的2D-3D图像配准

## 开发阶段

### 第一阶段: 2D图像处理 (当前阶段)
- [x] 项目结构搭建
- [ ] 图像预处理和分割
- [ ] 轮廓特征提取
- [ ] 基础可视化功能

### 第二阶段: 3D模型处理 (计划中)
- [ ] 3D模型加载和处理
- [ ] 多角度投影生成
- [ ] 投影轮廓提取

### 第三阶段: 配准算法 (计划中)
- [ ] CPD点集配准算法
- [ ] "一对多"配准策略
- [ ] 配准精度评估

### 第四阶段: 实时系统 (计划中)
- [ ] 实时图像采集
- [ ] 性能优化
- [ ] 用户界面

## 快速开始

### 环境设置
```bash
# 创建虚拟环境
python -m venv tooth_registration_env
source tooth_registration_env/bin/activate  # Linux/Mac
# 或 tooth_registration_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行示例
```bash
# 处理单张图片
python scripts/process_single_image.py --input data/raw/microscope_images/sample.jpg

# 批量处理
python scripts/batch_process.py --input_dir data/raw/microscope_images/

# 运行演示
python scripts/demo.py
```

## 项目结构
```
├── src/core/           # 核心算法模块
├── data/              # 数据目录
├── tests/             # 测试代码
├── notebooks/         # 实验笔记
├── scripts/           # 执行脚本
└── output/            # 输出结果
```

## 算法原理
本项目基于以下核心算法：
1. **自适应高斯滤波**: 根据图像局部特征调整滤波窗口
2. **改进Canny边缘检测**: 优化双阈值检测和非极大值抑制
3. **CPD点集配准**: 相干点漂移算法实现精确配准

## 开发日志
- 2024-01-XX: 项目初始化，完成基础结构搭建
- 更多记录请查看 `docs/development_log.md`

## 参考文献
张嘉伟. 基于内窥镜与CT图像融合的机器人辅助根管治疗术中导航研究[D]. 哈尔滨理工大学, 2024.