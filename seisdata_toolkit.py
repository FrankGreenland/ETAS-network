import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dtime
import geopy.distance
from sympy import symbols, solve, evalf


def log_bin_geometric_series(a1, q, kmax):
    # 2024-10-13
    # 为了进行log-binning 根据起始与q计算分几个格子
    n = symbols('n')
    series_sum = a1 * (1 - np.power(q, n)) / (1 - q)
    n_solved = solve(series_sum <= kmax).args[1]
    return n_solved


def log_bin_q(n, kmax, a1):
    q = np.power(kmax / a1, 1/n)
    return q


def logarithmic_binning(ks, a1, q, kmax, n):
    # 2024-10-15
    # 进行log-binning进行计算
    # 还没完成，预计不完成了，通过之前转换为线性后再操作
    ks = sorted(ks)
    n = 12
    q = log_bin_q(n, kmax, a1)
    bins_edge = np.zeros(n + 1)
    bins = np.logspace(0, n, n+1, base=q)
    bins_centers = np.sqrt(bins[:-1] * bins[1:])
    arr = np.histogram(ks, bins)
    arr__ = arr[0] / (bins[1:] - bins[:-1])
    plt.loglog(bins_centers, arr__[:], marker='o')
    plt.show()
    n = log_bin_geometric_series(a1, q, kmax)
    n = float(n)
    aaa = np.logspace(np.log(a1)/np.log(q), int(np.ceil(np.log(kmax) / np.log(q))), num=int(np.round(n)), base=q)
    plt.hist(ks, range=bins, log=True)
    return


def draw_loglog(x_fit, y_fit, xx, yy):
    # 2024-10-13
    # 绘制loglog图像参数都要改,具体使用时参数太多了，直接在这里手动改吧
    # 由瑞敏的程序改过来
    ticks_size = 13
    label_size = 15
    figure = plt.figure(figsize=(15, 12), dpi=300)
    figure_ax1 = figure.add_axes((0.07, 0.55, 0.40, 0.40))  # Sets the position of the layer on the canvas
    figure_ax2 = figure.add_axes((0.55, 0.55, 0.40, 0.40))  # Sets the position of the layer on the canvas
    figure_ax3 = figure.add_axes((0.07, 0.08, 0.40, 0.40))
    figure_ax4 = figure.add_axes((0.55, 0.08, 0.40, 0.40))
    figure_ax1.annotate('(a)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center",
                        va="center")
    figure_ax2.annotate('(b)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center",
                        va="center")
    figure_ax3.annotate('(c)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center",
                        va="center")
    figure_ax4.annotate('(d)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center",
                        va="center")
    # y_fit_ = y_fit/np.sum(y_fit)
    yy_ = yy/np.sum(yy)
    img21, = figure_ax2.loglog(x_fit, y_fit, color='black', linestyle='dashdot')
    img22 = figure_ax2.scatter(xx, yy_, color='black', marker='+', s=200, linewidths=2)
    figure_ax2.set_xlim([0, 1e2])
    # figure_ax2.set_ylim([1e-2, 5e3])
    figure_ax2.set_ylim([1e-5, 1])
    figure_ax2.legend([img21, img22], ['fit line', 'distribution density'])

    figure_ax1.set_xlabel(r'$d_{2D}$ (km)', fontsize=label_size)
    figure_ax1.set_ylabel(r'$\left(d_{3D}\ /\ d_{2D}\right)^3$', fontsize=label_size)
    figure_ax1.tick_params(labelsize=ticks_size)
    figure_ax1.grid(which='major', linestyle='--')
    figure_ax1.tick_params(axis='x', labelrotation=0)
    figure_ax1.tick_params(axis='y', labelrotation=0)
    figure_ax1.set_axisbelow(True)
    figure_ax1.minorticks_on()

    figure_ax2.set_xlabel(r'$\overline{Q_{3D}}$ $(m^3 s^{-1})$', fontsize=label_size)
    figure_ax2.set_ylabel(r'$\overline{Q_{2D}}^2$ $(m^4 s^{-2})$', fontsize=label_size)
    figure_ax2.tick_params(labelsize=ticks_size)
    figure_ax2.grid(which='both', linestyle=':')
    figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
    figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
    figure_ax2.set_axisbelow(True)
    figure_ax2.minorticks_on()

    figure_ax3.set_xlabel('Model Name', fontsize=label_size)
    figure_ax3.set_ylabel('Temperature (K)', fontsize=label_size)
    figure_ax3.tick_params(labelsize=ticks_size)
    figure_ax3.grid(which='major', linestyle='--')
    figure_ax3.tick_params(axis='x', labelrotation=-22.8, grid_linestyle='-')
    figure_ax3.tick_params(axis='y', labelrotation=0)
    figure_ax3.set_axisbelow(True)
    figure_ax3.minorticks_off()

    figure_ax4.set_xlabel(r'$T_{3D} - T_0$ (K)', fontsize=label_size)
    figure_ax4.set_ylabel(r'$T_{2D} - T_0$ (K)', fontsize=label_size)
    figure_ax4.tick_params(labelsize=ticks_size)
    figure_ax4.grid(which='major', linestyle='--')
    figure_ax4.tick_params(axis='x', labelrotation=0)
    figure_ax4.tick_params(axis='y', labelrotation=0)
    figure_ax4.set_axisbelow(True)
    figure_ax4.minorticks_on()

    plt.show()
    return 1


def max_min_scale(arr, mmax, mmin):
    # 2024-10-12
    # 将数据归一化到范围里
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    std = (arr - arr_min) / (arr_max - arr_min)
    arr_scaled = std * (mmax - mmin) + mmin
    return arr_scaled

# 2024/10/7
# 将各种类型的dict转为array
# 目前将一列的转换，两列的转换
def dict2array(dic):
    if dict == type(dic):
        print(dic)
        jj = iter(dic.values())
        ii = next(jj)
        if float == type(ii):
            arr = np.zeros((len(dic), 1), float)
        else:
            arr = np.zeros((len(dic), len(ii)), float)
        ii = 0
        for kk in dic.values():
            arr_kk = np.array(kk)
            arr[ii] = arr_kk
            ii = ii + 1
        return arr
    else:
        print('非dict类型')
        return 0


def count_years(data):
    # 统计每年多少个地震
    start_year = data[0, 0]
    end_year = data[len(data) - 1, 0]
    num_years = int(np.around(end_year - start_year + 1, decimals=0))
    counts = np.zeros(num_years)
    k = 0
    for i in range(len(data)):
        if start_year != data[i, 0]:
            start_year = data[i, 0]
            # print(i)
            # print(start_year)
            counts[k] = i
            k = k + 1
            # print(k)
    counts[num_years - 1] = len(data)
    for i in range(len(counts) - 1, 0, -1):
        counts[i] = counts[i] - counts[i - 1]
    return counts


def count_mag(data):
    # 统计地震等级
    start_mag = -0.1
    end_mag = 9
    delta_mag = 0.1
    mag_len = (end_mag - start_mag) / delta_mag


def select_m(es, minM=0.1):
    done = es
    i = len(es)
    for i in range(len(done)):
        if done[i] + 0.001 >= minM:
            break
    ans = done[i:]
    return ans


def mean_m(events):
    if len(events) == 0:
        meanM = np.nan
    else:
        meanM = 0
        for i in range(len(events)):
            meanM = meanM + events[i]
        meanM = meanM / len(events)
    return meanM


def caculate_b(es, year=np.NaN):
    # 计算b的值
    Mmax = 8
    step = 0.1
    mN = Mmax / step
    M = np.zeros(int(mN) + 1)
    Mx = np.zeros(int(mN) + 1)
    Mc = np.zeros(int(mN))
    _b = np.zeros(int(mN), float)
    a = np.log10(np.e)
    for i in range(len(es)):
        # print(es[i, 4] / step - 1)
        mp = int(np.round(es[i, 6] * 10))
        # print(i, es[i, 4], mp)
        M[mp] = M[mp] + 1
    for i in range(len(Mx)):
        Mx[i] = (i + 1)
    for i in range(len(M)):
        if M[i] == 0:
            M[i] = np.NaN
        else:
            M[i] = np.log10(M[i])
    plt.figure()
    plt.scatter(Mx, M)
    plt.title("F(m)" + str(year))
    plt.savefig("figures\\{}.png".format("震级分布" + str(year)))
    plt.close()
    mmm = es[:, 6]
    m_index = np.lexsort((mmm,))
    done = mmm.T[m_index].T
    for j in range(35):
        # print(j)
        Mc[j] = j * step + 0.1
        cutm = select_m(done, minM=Mc[j])
        mean = mean_m(cutm)
        _b[j] = a / (mean - Mc[j] + 0.000001)
    plt.figure()
    plt.title("b" + str(year))
    plt.scatter(Mc, _b)
    plt.savefig("figures\\{}.png".format("b" + str(year)))
    plt.close()
    return _b


def G_Rline(es, titles="G-R"):
    Mmax = 8
    step = 0.1
    mN = int(Mmax / step)  # 0号代表震级0.1
    N_m = np.zeros((mN), 'int')
    logN = np.zeros((mN), 'float')
    for i in range(mN):
        N_m[i] = len(es[es[:, 6] > (i + 1) * 0.1])
    x = np.linspace(0.1, 8, mN)
    for i in range(len(N_m)):
        if N_m[i] != 0:
            logN[i] = np.log(N_m[i])
        else:
            logN[i] = np.NaN
    plt.figure()
    plt.xticks = np.linspace(0.1, 8, mN)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(titles)
    plt.xlabel('面波震级Mc')
    plt.ylabel("lgN")
    plt.scatter(x, logN)
    plt.savefig("figures\\{}.png".format("G-R" + titles))
    plt.close()
    NN = np.zeros((mN))
    for i in range(mN):
        NN[i] = ((es[:, 6] * 10).astype('int') == (i + 1)).sum()
    plt.figure()
    plt.title(titles)
    plt.ylabel('频度N')
    plt.xlabel('面波震级Mc')
    plt.bar(x, NN)
    plt.savefig("figures\\{}.png".format("b" + titles))
    plt.close()
    return N_m
