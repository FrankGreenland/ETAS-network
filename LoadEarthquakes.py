import pandas as pd
import numpy as np


def load_etas(region="shanxi", thm=2.5):
    # 读取放入etas程序的地震目录
    if region == 'chuandian':
        catalogs = pd.read_csv("data/" + region + "/" + region + ".etas", sep='\\s+', header=None)
        catalogs = catalogs[catalogs[3] >= thm]
    else:
        catalogs = pd.read_csv("data/" + region + "/" + region + ".etas", sep=' ',
                               names=["No", "long", "lat", "M", "time", "depth", "year", "month", "day"])
        catalogs = catalogs[catalogs["M"] >= thm]
    return catalogs


def load_pmatr(region="shanxi", p1=1e-2):
    # 将etas的相互触发结果加载
    pmatr = pd.read_csv("data/" + region + "/pmatr.dat",
                        header=None, sep='\\s+',
                        dtype={'column1': np.int8, 'column2': np.int8, 'column3': np.float32}
                        )
    # 删除与自己的连接，并转化为numpy数组
    selected_data = np.delete(pmatr.values,
                              np.where(pmatr.values[:, 0].astype(int) - pmatr.values[:, 1] == 0), axis=0)
    deleted_data = np.delete(selected_data, np.where(selected_data[:, 2] <= p1), axis=0)
    deleted_data[:, [0, 1]] = deleted_data[:, [1, 0]]  # 交换前两列
    if region != "shanxi":
        # 跑etas的时候第一个地震没计算上，不知道原因，可能为起始时间的问题
        deleted_data[:, 0:2] = deleted_data[:, 0:2] + 1
    return deleted_data
