# 专门为了画图的小函数放在这里了
import matplotlib.pyplot as plt
import numpy as np

def degree_distribution(arr, sbin=10, bbin=10, method='log'):
    # 由于度分布是离散的，log-binning在数较小的时候不好用
    # liner-bin 在尾部也不好用
    # 所以前部分直接用每个度来
    # 后面用log-binning
    # method 如果是 'log' 则为log-binning, 如果是 'liner' 则为线性分箱
    arr = sorted(arr)
    avg = np.sqrt(np.max(arr) * np.min(arr))
    arr_s = np.array(arr)[arr < avg]  # 线性区域
    arr_b = np.array(arr)[arr >= avg]  # log 区域
    counts, bins_edges = np.histogram(arr_s, bins=sbin)
    xx = bins_edges[:len(bins_edges) - 1]
    xx = 0.5 * (xx[1] - xx[0]) + xx
    yy = counts / sum(counts)
    plt.plot(xx, yy)
    plt.show()
    return 1


