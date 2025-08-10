import numpy as np
import matplotlib.pyplot as plt

# 加载label.npy文件
label = np.load('../../label.npy')  # shape: [D, H, W]，通常是 Z, Y, X
label = np.unque

# 打印shape确认维度
print("Label shape:", label.shape)

# 选择切片位置
z = label.shape[0] // 2  # axial 中间层
y = label.shape[1] // 2  # coronal 中间层
x = label.shape[2] // 2  # sagittal 中间层

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Axial view (平面图)
axes[0].imshow(label[z, :, :], cmap='nipy_spectral')
axes[0].set_title(f'Axial Slice z={z}')

# Coronal view (冠状面)
axes[1].imshow(label[:, y, :], cmap='nipy_spectral')
axes[1].set_title(f'Coronal Slice y={y}')

# Sagittal view (矢状面)
axes[2].imshow(label[:, :, x], cmap='nipy_spectral')
axes[2].set_title(f'Sagittal Slice x={x}')

plt.tight_layout()
plt.show()
