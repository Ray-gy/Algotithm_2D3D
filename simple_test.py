import numpy as np
import matplotlib.pyplot as plt
import cv2

# 创建一个简单的例子来说明原理
def create_simple_example():
    """创建一个包含边缘和噪声的简单图像"""
    # 创建一个黑白边界的图像
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255  # 右半部分是白色
    
    # 添加随机噪声
    noise = np.random.randint(-30, 30, img.shape)
    img_noisy = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return img, img_noisy

# 手动实现简化的双边滤波来展示原理
def simple_bilateral_demo(img, center_row, center_col, sigma_color, sigma_space):
    """
    演示双边滤波在特定像素点的计算过程
    """
    height, width = img.shape
    window_size = 5
    half_window = window_size // 2
    
    center_value = img[center_row, center_col]
    print(f"处理像素位置: ({center_row}, {center_col}), 中心像素值: {center_value}")
    print("=" * 60)
    
    total_weight = 0
    weighted_sum = 0
    
    print("邻域像素分析:")
    print("位置\t像素值\t颜色差\t空间距离\t颜色权重\t空间权重\t总权重")
    print("-" * 70)
    
    for i in range(max(0, center_row - half_window), 
                   min(height, center_row + half_window + 1)):
        for j in range(max(0, center_col - half_window),
                       min(width, center_col + half_window + 1)):
            
            pixel_value = img[i, j]
            
            # 计算颜色差异
            color_diff = abs(int(pixel_value) - int(center_value))
            
            # 计算空间距离
            spatial_dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            
            # 计算权重
            color_weight = np.exp(-(color_diff**2) / (2 * sigma_color**2))
            spatial_weight = np.exp(-(spatial_dist**2) / (2 * sigma_space**2))
            total_pixel_weight = color_weight * spatial_weight
            
            print(f"({i},{j})\t{pixel_value}\t{color_diff}\t{spatial_dist:.1f}\t\t{color_weight:.3f}\t\t{spatial_weight:.3f}\t\t{total_pixel_weight:.3f}")
            
            weighted_sum += pixel_value * total_pixel_weight
            total_weight += total_pixel_weight
    
    final_value = weighted_sum / total_weight if total_weight > 0 else center_value
    print(f"\n最终结果: {weighted_sum:.1f} / {total_weight:.3f} = {final_value:.1f}")
    
    return final_value

# 创建测试图像
original, noisy = create_simple_example()

# 显示图像
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title('原始图像（黑白边界）')
plt.axis('off')

plt.subplot(1, 3, 2)  
plt.imshow(noisy, cmap='gray')
plt.title('添加噪声后')
plt.axis('off')

# 应用双边滤波
bilateral_result = cv2.bilateralFilter(noisy, 9, 50, 50)

plt.subplot(1, 3, 3)
plt.imshow(bilateral_result, cmap='gray')
plt.title('双边滤波结果')
plt.axis('off')

plt.tight_layout()
plt.show()

print("双边滤波工作原理演示")
print("=" * 50)

# 选择边界附近的一个像素点来分析
center_row, center_col = 50, 48  # 黑白边界附近

print("\n情况1: sigmaColor=10 (只处理相似颜色)")
simple_bilateral_demo(noisy, center_row, center_col, sigma_color=10, sigma_space=2)

print("\n" + "="*70)
print("\n情况2: sigmaColor=100 (处理差异较大的颜色)")
simple_bilateral_demo(noisy, center_row, center_col, sigma_color=100, sigma_space=2)

print("\n" + "="*70)
print("关键理解点:")
print("1. 双边滤波同时考虑空间距离和颜色相似度")
print("2. sigmaColor小 → 只有颜色相近的像素参与平均 → 保护边缘")
print("3. sigmaColor大 → 颜色差异大的像素也参与平均 → 更强平滑")
print("4. 在边界处，黑色像素(0)和白色像素(255)颜色差异很大")
print("5. 当sigmaColor小时，黑白像素互不影响，保持边缘清晰")
print("6. 当sigmaColor大时，黑白像素会相互影响，边缘变模糊")

# 对比不同参数的效果
plt.figure(figsize=(15, 10))

sigma_color_values = [10, 50, 150]
sigma_space = 2

for i, sigma_c in enumerate(sigma_color_values):
    result = cv2.bilateralFilter(noisy, 9, sigma_c, sigma_space)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(f'sigmaColor={sigma_c}')
    plt.axis('off')
    
    # 显示边界处的像素值剖面
    plt.subplot(2, 3, i+4)
    plt.plot(result[50, :], label=f'sigmaColor={sigma_c}')
    plt.plot(original[50, :], '--', alpha=0.5, label='原始边界')
    plt.xlabel('像素位置')
    plt.ylabel('像素值')
    plt.title(f'第50行的像素值变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()