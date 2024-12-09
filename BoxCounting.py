import numpy as np
import matplotlib.pyplot as plt

# 在图像中创建方形
image = np.zeros((512, 512))
image[200:300, 200:300] = 1


def fractal_box_count(image, min_size=4, max_size=None, step=2):
    """
    计算分形盒计数
    """
    shape = image.shape
    if max_size is None:
        max_size = min(shape[0], shape[1])
    x, y = np.where(image == 1)
    points = np.column_stack((x, y))
    count = []
    scales = []
    for size in range(min_size, max_size + 1, step):
        scales.append(size)
        boxes = np.ceil(shape[0] / size) * np.ceil(shape[1] / size)
        if boxes == 0:
            count.append(0)
            continue
        counts = np.zeros((int(np.ceil(shape[0] / size)), int(np.ceil(shape[1] / size))))
        for point in points:
            i, j = np.floor(point / size).astype(int)
            counts[i, j] = 1
        count.append(np.sum(counts > 0))
    return np.array(count), np.array(scales)


# 计算分形盒计数
count, scales = fractal_box_count(image)

# 绘制线性图
plt.plot(np.log(scales), np.log(count), 'bo-')

# 添加坐标轴标签
plt.xlabel('log(网格尺度)')
plt.ylabel('log(盒子数)')
plt.title('分形盒计数')
plt.show()
