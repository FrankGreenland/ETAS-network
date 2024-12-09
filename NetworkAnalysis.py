import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from geographiclib.geodesic import Geodesic
from scipy.optimize import curve_fit
import scipy.stats as sta
import NetworkGeneration as ng

from NetworkGeneration import graphGenerate
from seisdata_toolkit import dict2array, max_min_scale

# 绘图设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300


def list2array(aim_list):
    # 实际上是dict转array
    if type(aim_list) == list:
        array = np.array(aim_list)
    else:
        array = np.zeros(len(aim_list), np.float64)
        i = 0
        for ii in aim_list.values():
            array[i] = ii
            i = i + 1
    return array


def drawgraph(G, region='shanxi'):
    if region == 'shanxi':
        plt.rcParams['figure.figsize'] = (8, 14)
    elif region == 'INGV':
        plt.rcParams['figure.figsize'] = (12, 14)  # Italy
    else:
        plt.rcParams['figure.figsize'] = (15, 11)  # southern california
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # 尝试画G图
    pos1 = nx.get_node_attributes(G, 'position')
    # drawDensity = 10
    # for key in pos1.keys():
    #    pos1[key] = (pos1[key][0] * drawDensity, pos1[key][1] * drawDensity)
    cmap = plt.cm.viridis
    node_size = 50
    edges_color = list2array(nx.get_edge_attributes(G, 'weight'))
    edges = nx.draw_networkx_edges(G, pos1,  width=0.3, arrowsize=3, node_size=node_size,
                           edge_color=edges_color, edge_cmap=cmap)
    nodes = nx.draw_networkx_nodes(G, pos=pos1, node_size=node_size, alpha=0.8)
    # nx.draw(G, pos1, node_size=60, alpha=0.8, with_labels=True, font_size=6, width=0.5)
    plt.title('有向图')

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edges_color)
    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax, fraction=0.05)
    plt.show()

    # 去除自环的图
    G2 = G.copy()
    G2.remove_edges_from(nx.selfloop_edges(G2))
    edges_color = list2array(nx.get_edge_attributes(G2, 'weight'))
    edges = nx.draw_networkx_edges(G2, pos1,  width=0.3, arrowsize=3, node_size=node_size,
                           edge_color=edges_color, edge_cmap=cmap)
    nodes = nx.draw_networkx_nodes(G2, pos=pos1, node_size=node_size, alpha=0.8)
    # nx.draw(G, pos1, node_size=60, alpha=0.8, with_labels=True, font_size=6, width=0.5)
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edges_color)
    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax, fraction=0.05)
    plt.show()
    return


def map_scale(data, aim_min, aim_max):
    # 将数据归一化到指定的区间
    d_min = np.min(data)
    d_max = np.max(data)
    return aim_min + (aim_max - aim_min) / (d_max - d_min) * (data-d_min)


def draw_properties(g, prop, factor=500, title_name='title', font_size=20, font_color='k', region='shanxi'):
    if region == 'shanxi':
        plt.rcParams['figure.figsize'] = (8, 14)
    elif region == 'INGV':
        plt.rcParams['figure.figsize'] = (12, 14)  # Italy
    elif region == 'chuandian':
        plt.rcParams['figure.figsize'] = (8, 12)  # 川滇
    else:
        plt.rcParams['figure.figsize'] = (15, 11)  # southern california

    # 绘制每个node的大小
    size = list2array(prop)
    cmap = plt.cm.plasma
    edges_color = list2array(nx.get_edge_attributes(g, 'weight'))
    if np.min(size) == 0:
        size_s = size + 0.01
    else:
        size_s = size
    pos = nx.get_node_attributes(g, 'position')
    # edges
    #edges = nx.draw_networkx_edges(g, pos,  width=0.3, arrowsize=3, node_size=size_s*factor,
    #                       edge_color=edges_color, edge_cmap=plt.cm.Reds )
    edges = nx.draw_networkx_edges(g, pos,  width=0.3, arrowsize=3, node_size=size_s*factor,
                                   edge_color='k', alpha=0.4)
    # 绘制图的顶点
    nodes = nx.draw_networkx_nodes(g, pos=pos, node_size=size_s * factor, node_color=size_s, cmap=plt.cm.plasma)
    # nx.draw_networkx_nodes(g, pos=pos, node_size=size * factor, node_color= "indigo", cmap=plt.cm.plasma, alpha=0.7)
    # 绘制图上的标注
    # nx.draw_networkx_labels(g, pos=pos, labels=prop, font_size=font_size, font_color=font_color, font_weight='normal')
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(size)
    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax, fraction=0.05)
    # plt.title(title_name + ' liner')
    plt.show()

    '''
    size_log = map_scale(size, aim_min=np.power(1.01, 10), aim_max=np.power(1.01, factor * 0.25))
    size_log_pow = np.log(size_log) / np.log(1.01)
    nx.draw_networkx_nodes(g, pos=pos, node_size=size_log_pow, node_color=size, cmap=plt.cm.Reds_r)
    plt.title(title_name + 'log')
    plt.show()
    '''
    return


def scaleFreeDraw(ks, bins=8, beizhu='无权'):
    # 绘制无边值的图, x轴为正常轴，绘制节点的度分布
    plt.rcParams['figure.figsize'] = (6, 6)
    degree_sequence = ks
    fig, ax = plt.subplots()
    ax.bar(*np.unique(degree_sequence, return_counts=True))
    ax.set_title('Degree histogram' + beizhu)
    ax.set_xlabel('Degree')
    ax.set_ylabel('# of Nodes')
    plt.show()

    # sns.displot(degree_sequence, kde=True)
    # plt.show()

    plt.hist(degree_sequence, bins=bins + 2, edgecolor='black')
    plt.title('Degree histogram' + beizhu)
    plt.show()

    plt.hist(degree_sequence, bins=bins, edgecolor='black')
    plt.title('Degree histogram' + beizhu)
    plt.show()

    counts, bin_edges = np.histogram(degree_sequence, bins=bins)

    return counts, bin_edges


def scaleFreeLogDraw(ks, bins=20):
    counts, bins_edges = np.histogram(ks, bins=bins)
    # 绘制无边值图，x轴为对数轴
    # counts=counts[1:]
    # bins_edges=bins_edges[1:]
    xx = bins_edges[:len(bins_edges) - 1]
    xx = 0.5 * (xx[1] - xx[0]) + xx

    yy = counts / sum(counts)
    plt.plot(xx, yy)
    plt.show()


    plt.loglog(xx, yy, 'o')
    plt.show()
    return xx, yy


def scaleFreeLogPinDraw(GraphWeight, bins=20):
    # 以logarithmic binning绘制‘直方图’以消除重尾影响
    GraphWeight = np.array(GraphWeight)
    if np.sum(GraphWeight != 0) == 1:
        ksLog = np.log10(GraphWeight)
    else:
        ksLog = np.log10(GraphWeight[GraphWeight != 0])
    counts, bins_edges = np.histogram(ksLog, bins=bins)
    line = np.power(10, bins_edges)
    diEdges = line[1:] - line[:-1]
    yy = counts / diEdges
    yP = yy / sum(yy)
    xx = 0.5 * (bins_edges[1] - bins_edges[0]) + bins_edges
    xx = xx[:len(bins_edges) - 1]
    xx = np.power(10, xx)

    plt.loglog(xx, yP, 'o')
    plt.show()

    plt.stairs(yy, line)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('log bin hist')
    plt.show()

    kp, yp = fittinglinear(xx, yP)
    return xx, yP, kp, yp


# 拟合幂律
def func(x, c, a):
    return c * np.power(x, -1 * a)


def funclinear(x, c, a):
    return -1 * a * x + c

def fittinglinear(xx, yy):
    k = np.log10(xx)
    kprob = np.log10(yy)
    k2 = k[kprob != -1*np.inf]
    kprob2 = kprob[kprob != -1*np.inf]
    plt.scatter(k2, kprob2, color='b')
    popt, pcov = curve_fit(funclinear, k2, kprob2)
    y2 = [funclinear(i, popt[0], popt[1]) for i in k2]
    plt.plot(k2, y2, 'r--')
    plt.title('c:' + str(popt[0]) + '   a=' + str(popt[1]))
    plt.show()
    return np.power(10, k2), np.power(10, np.array(y2))


def fittingEx(xx, yy):
    k = xx
    probK = yy
    plt.scatter(k, probK, color='b')
    popt, pcov = curve_fit(func, k, probK)
    y2 = [func(i, popt[0], popt[1]) for i in k]
    plt.plot(k, y2, 'r--')
    plt.title('c:' + str(popt[0]) + '   a=' + str(popt[1]))
    plt.show()


    plt.loglog(xx, yy, 'o')
    # plt.loglog(k, y2, 'o')
    plt.show()

    return popt, pcov



def average_shortest_path_len(G):
    # 将含有多个组件的图拆开并计算平均长度
    GUndirected = G.to_undirected()  # 转化为无向图
    numComponents = nx.number_connected_components(GUndirected)  # 有n个组件
    isoNode = list(nx.isolates(GUndirected))
    GUndirected.remove_nodes_from(isoNode)  # 去除孤立点
    graphs = list(nx.connected_components(GUndirected))
    sumLen = 0
    sumNum = 0
    for i in graphs:
        subG = GUndirected.subgraph(i)
        sumLen = sumLen + subG.number_of_nodes() * nx.average_shortest_path_length(subG)
        sumNum = subG.number_of_nodes() + sumNum
    averageLen = sumLen / sumNum
    return averageLen


def cap_degree(graph, is_out=False, is_in=False, b=50):
    # 获得图的度分布
    G = graph
    GraphInDegree = sorted([d for j, d in G.in_degree(weight='weight')], reverse=True)
    GraphOutDegree = sorted([d for ii, d in G.out_degree(weight='weight')], reverse=True)
    if is_in == True:
        GraphWeight = np.array(GraphInDegree)
        plt.rcParams['figure.figsize'] = (6, 6)
        counts2, bins_edges2 = scaleFreeDraw(GraphWeight, bins=b, beizhu='加权图')
        k, probK = scaleFreeLogDraw(GraphWeight, bins=b)
        fittingEx(k, probK)
        fittinglinear(k, probK)
        # 绘制log bin 度分布
        scaleFreeLogPinDraw(GraphWeight, bins=b)

    if is_out == True:
        GraphWeight = np.array(GraphOutDegree)
        plt.rcParams['figure.figsize'] = (6, 6)
        counts2, bins_edges2 = scaleFreeDraw(GraphWeight, bins=b, beizhu='加权图')
        k, probK = scaleFreeLogDraw(GraphWeight, bins=b)
        fittingEx(k, probK)
        fittinglinear(k, probK)
        # 绘制log bin 度分布
        scaleFreeLogPinDraw(GraphWeight, bins=b)

    return GraphOutDegree, GraphInDegree


def clustering_random(degrees, count_zeros=False):
    # 输入度，以随机图的形式，计算C
    degrees = np.array(degrees)
    if not count_zeros:
        degrees = degrees[degrees != 0]
    n = len(degrees)
    mean_degree = sum(degrees) / n
    mean_square_degree = sum(degrees * degrees) / n
    mean_tri_degree = sum(degrees * degrees * degrees) / n
    c = np.power((mean_square_degree - mean_degree), 2) / mean_tri_degree / n
    return c


# !!!
region = 'southern_california'
nLen = 4
mProb = 1e-3
thDistance = 5 + nLen/2  # 默认为5，如果想去掉这个限制，可以输入一个很大的数，比如10000

thm = 2.4  # shanxi thm=2.5 |||| INGV  thm=2.9|||| Chuandian  thm=2.8||| southern california thm=2.4  很重要，不然会有bug
G, boxes, event_id2node_id, pmatr, catalog, boxes_coord, all_box = graphGenerate(region, nLen, mProb, thDistance, thm)
regionBorder = ng.get_border_file(region)  # 获取研究区域的边界值
longDistance, latDistance = ng.border_length(regionBorder)  # 计算研究区域长度跨度
nyLen = (regionBorder[3] - regionBorder[2]) * nLen / latDistance  # 一个格子的经纬度跨度
nxLen = (regionBorder[1] - regionBorder[0]) * nLen / longDistance

# 去掉自环的图
G2 = G.copy()
G2.remove_edges_from(nx.selfloop_edges(G2))
# cap_degree(G2, is_in=True, is_out=True, b=50)  # 计算出度与入度的distribution

plt.clf()
drawgraph(G, region)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
counts, bins_edges = scaleFreeDraw(degree_sequence, 50, '无权')
k, probK = scaleFreeLogDraw(degree_sequence, 30)
fittingEx(k, probK)
fittinglinear(k[1:], probK[1:])
scaleFreeLogPinDraw(degree_sequence, bins=30)
np.save('./plot_degree/' + region + '_degree_x.npy', k)
np.save('./plot_degree/' + region + '_degree_y.npy', probK)

# w的统计
weights = []
for _, __, weight in G.edges(data='weight'):
    weights.append(weight)
counts, bins_edges = scaleFreeDraw(weights, 20)
k, probK = scaleFreeLogDraw(weights, 20)
fittingEx(k, probK)
xline, yline = fittinglinear(k, probK)
xx, yy, _, _ = scaleFreeLogPinDraw(weights, bins=30)
np.save('./plot_degree/' + region + '_weight_x.npy', xx)
np.save('./plot_degree/' + region + '_weight_y.npy', yy)


# node strength
GraphInDegree = sorted([d for j, d in G.in_degree(weight='weight')], reverse=True)
GraphOutDegree = sorted([d for ii, d in G.out_degree(weight='weight')], reverse=True)
k, probK = scaleFreeLogDraw(GraphOutDegree, 30)
xx, yy, _, _ = scaleFreeLogPinDraw(GraphOutDegree, bins=30)
np.save('./plot_degree/' + region + '_strength_x.npy', xx)
np.save('./plot_degree/' + region + '_strength_y.npy', yy)



# 无向图
G3 = G.copy()
G3 = G3.to_undirected()

# 获得带权重的度数图
# node strength
GraphInDegree, GraphOutDegree = cap_degree(G, is_in=True, is_out=True, b=30)  # 计算出度与入度的distribution
draw_properties(G2, [d for ii, d in G.out_degree(weight='weight')], factor=50, title_name='out degree', region=region)

# 社区detection community
degreeCentrality = nx.degree_centrality(G2)
resolution_para = 1
communities = nx.community.louvain_communities(G2, weight='weight', resolution=resolution_para, seed=1)
print(f"{len(communities)}")
# 把社区detection绘制出来
pos = nx.get_node_attributes(G, 'position')
colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
size_factor = 1200
min_size = 10
for nodes, clr in zip(communities, colors):
    size = []
    for node in nodes:
        n_size = degreeCentrality[node]
        size.append(n_size)
    if len(size) >= min_size:
        nx.draw_networkx_nodes(G2, pos=pos, nodelist=nodes, node_size=np.array(size)*size_factor, node_color=clr)
nx.draw_networkx_edges(G2, pos=pos, width=0.3, arrowsize=3, alpha=0.3)
plt.show()
modularity_weight = nx.community.modularity(G2, communities, weight="weight", resolution=resolution_para)
generator = nx.community.louvain_partitions(G2, weight='weight', seed=1, resolution=resolution_para)
coms = iter(generator)
aaa = next(coms)
communities = aaa
modularity_weight = nx.community.modularity(G2, communities, weight="weight", resolution=resolution_para)

# 将过小的community去除
min_size_community = 2
community_arr = np.array(communities)
big_community = []
for comm in community_arr:
    if len(comm) < min_size_community:
        print(comm)
        print(len(comm))
    else:
        big_community.append(comm)

ng.drew_community(all_box, big_community, boxes, nxLen, nyLen, latDistance, longDistance, regionBorder, fpath='GMT_plot_community/')
# ng.drew_community_region(region, all_box, big_community, boxes, nxLen, nyLen, latDistance, longDistance, regionBorder, fpath='GMT_plot_community_combine/')

def same_comm_count(G2, pmatr, region, resolution_para=1, thm=5, thp=0.1):
    # 将网络依次生成不同层级的partitions
    # 计算不同层级的partition，输入给绘图软件
    # 选取某一震级以上的地震，计算这些地震的子代的子代地震的set
    # 在这些set中，是否都是在同一个community内
    fpath = 'plot_community_percent'
    f3 = open(f"{fpath}/{region}_percent.dat", 'w')
    generator = nx.community.louvain_partitions(G2, weight='weight', seed=1, resolution=resolution_para)
    k_comm = 0
    t_catalog = catalog[catalog[:, 3] >= thm]
    print(f'{thm}级以上地震共{len(t_catalog)}个')
    for communities in generator:
        print(f"现在是第{k_comm}个层级")
        sum_same_com = 0
        offsprings_per = np.zeros(len(t_catalog))  # 每个大地震的子代在同一个community的占比
        offsprings_num = np.zeros(len(t_catalog))  # 每个大地震的子代数量
        i = 0
        for event in t_catalog:
            # print(f"现在是第{i}个大地震")
            offsprings = ng.whole_offspring(event[0], pmatr, thp=thp)  # 找到子代
            offsprings_boxes_id = np.array(event_id2node_id[offsprings - 1])  # 子代的box id
            event_node = event_id2node_id[event[0].astype(int) - 1]  # 事件的box id
            offsprings_num[i] = len(offsprings)  # 对于event其子代的数量
            for comm in communities:  # 找到对应的事件
                comm_arr = np.array(list(comm))
                if event_node in comm_arr:
                    break
            offsprings_in_same_community = 0
            for offspring_box in offsprings_boxes_id:  # 循环判断每个子代是不是在一个community里面
                if offspring_box in comm_arr:
                    offsprings_in_same_community = offsprings_in_same_community + 1
            sum_same_com = sum_same_com + offsprings_in_same_community
            offsprings_per[i] = offsprings_in_same_community / offsprings_num[i]
            # print(f"子代有{offsprings_in_same_community}个地震事件在同一个community里面")
            # print(f"子代在同一个community的比例{offsprings_per[i]}")

            i = i + 1
        sum_all_offsprings = np.sum(offsprings_num)  # 所有的子代的数量
        all_percent = sum_same_com / sum_all_offsprings  # 所有子代中，在同一community的比例
        print(f"子代数量{sum_all_offsprings}")
        print(f"子代在同一社区里的比例为{all_percent}")
        f3.write(f"{all_percent}\n")
        k_comm = k_comm + 1
    print(f"共{k_comm}个层级")
    f1 = open(f"{fpath}/{region}_offspring_num.dat", 'w')
    f1.write(f"{sum_all_offsprings}")
    f2 = open(f"{fpath}/{region}_level_num.dat", "w")
    f2.write(f"{k_comm}")
    f1.close()
    f2.close()
    f3.close()
    return 1


def same_comm_count_resolution(G2, pmatr, region, resolution_para=np.array([0.01, 0.05, 0.1, 0.5, 1]), thm=5, thp=0.1):
    # 根据不同的resolution_para生成community
    # 选取某一震级以上的地震，计算这些地震的子代的子代地震的set
    # 在这些set中，是否都是在同一个community内
    fpath = 'plot_community_resolution'
    f3 = open(f"{fpath}/{region}_percent.dat", 'w')
    f2 = open(f"{fpath}/{region}_community_num.dat", "w")
    t_catalog = catalog[catalog[:, 3] >= thm]
    print(f'{thm}级以上地震共{len(t_catalog)}个')
    for reso in resolution_para:
        print(f"现在的gamma为{reso}")
        sum_same_com = 0
        communities = nx.community.louvain_communities(G2, weight='weight', resolution=reso, seed=1)
        print(f"共有{len(communities)}个communities")
        f2.write(f"{len(communities)}\n")
        offsprings_per = np.zeros(len(t_catalog))  # 每个大地震的子代在同一个community的占比
        offsprings_num = np.zeros(len(t_catalog))  # 每个大地震的子代数量
        i = 0
        for event in t_catalog:
            # print(f"现在是第{i}个大地震")
            offsprings = ng.whole_offspring(event[0], pmatr, thp=thp)  # 找到子代
            offsprings_boxes_id = np.array(event_id2node_id[offsprings - 1])  # 子代的box id
            event_node = event_id2node_id[event[0].astype(int) - 1]  # 事件的box id
            offsprings_num[i] = len(offsprings)  # 对于event其子代的数量
            for comm in communities:  # 找到对应的事件
                comm_arr = np.array(list(comm))
                if event_node in comm_arr:
                    break
            offsprings_in_same_community = 0
            for offspring_box in offsprings_boxes_id:  # 循环判断每个子代是不是在一个community里面
                if offspring_box in comm_arr:
                    offsprings_in_same_community = offsprings_in_same_community + 1
            sum_same_com = sum_same_com + offsprings_in_same_community
            offsprings_per[i] = offsprings_in_same_community / offsprings_num[i]
            # print(f"子代有{offsprings_in_same_community}个地震事件在同一个community里面")
            # print(f"子代在同一个community的比例{offsprings_per[i]}")

            i = i + 1
        sum_all_offsprings = np.sum(offsprings_num)  # 所有的子代的数量
        all_percent = sum_same_com / sum_all_offsprings  # 所有子代中，在同一community的比例
        print(f"子代数量{sum_all_offsprings}")
        print(f"子代在同一社区里的比例为{all_percent}")
        f3.write(f"{all_percent}\n")
    print('\n')
    f1 = open(f"{fpath}/{region}_offspring_num.dat", 'w')
    f1.write(f"{sum_all_offsprings}")
    f1.close()
    f3.close()
    return 1


resolution = 0.1
thm = 5
thp = 0.1
same_comm_count(G2, pmatr, region, resolution_para=resolution, thm=thm, thp=thp)
same_comm_count_resolution(G2, pmatr, region, thm=thm, thp=thp)

boxes_with_offspring = event_id2node_id[offsprings.astype(int)-1]
boxes = np.array(boxes)
boxes_pos = boxes[np.array(list(set(boxes_with_offspring))).astype(int)]
boxes_position_coord = ng.exchange_distance2coordinate(boxes_pos, latDistance, longDistance,
                                                    regionBorder)  # 将boxes转化为经纬度
ng.drew_box(all_box, boxes_position_coord, nxLen, nyLen)

# 创建邻接矩阵
A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
A = A.todense()
plt.rcParams['figure.figsize'] = (20, 20)
plt.matshow(A)
plt.title('邻接矩阵A')
plt.colorbar(fraction=0.04)
plt.show()


def againstk(G, coef, coefname='coef'):
    coef_list = np.array(list(coef.values()))
    node_degree = np.array([d for _, d in G.degree(coef.keys())])
    k2k = np.vstack((node_degree, coef_list)).T
    k2k_dataframe = pd.DataFrame(k2k, columns=['k', coefname])
    sns.jointplot(k2k_dataframe, x='k', y=coefname)
    plt.show()
    return k2k


def average_against_k(coef, k, bins=10):
    plt.rcParams['figure.figsize'] = (7, 7)
    num, edges = np.histogram(k, bins=bins)
    amk = np.zeros(len(num))
    for i in range(len(num)):
        if num[i] != 0:
            amk[i] = np.sum(coef[np.logical_and(k > edges[i], k <= edges[i + 1])]) / num[i]
        else:
            amk[i] = np.inf
    plt.xlabel('k')
    plt.ylabel('ClusteringCoefficient')
    plt.scatter(edges[:-1], amk)
    plt.show()
    return


# 分析clustering coefficient
clusteringCoeUnweighted = nx.clustering(G, weight=None)
clusteringCoeWeighted = nx.clustering(G, weight='weight')
draw_properties(G2, clusteringCoeWeighted, factor=1500, title_name='', region=region)
k2c = againstk(G, clusteringCoeUnweighted, 'ClusteringCoeUnweighted')
average_against_k(k2c[:, 1], k2c[:, 0], bins=20)
k2c = againstk(G, clusteringCoeWeighted, 'ClusteringCoeWeighted')
average_against_k(k2c[:, 1], k2c[:, 0], bins=20)


# small world
# 计算最小距离以确定small-world 特性
shortPath = average_shortest_path_len(G2)
# 全局聚焦系数
averageClusteringCoeUnweighted = nx.average_clustering(G, weight=None, count_zeros=False)
averageClusteringCoeWeighted = nx.average_clustering(G, weight='weight', count_zeros=False)
# 如果是该图的连接是随机的，则可以算得该顶点结构下的聚焦系数
# 计算随机图下的clustering coefficient
# 以比对该图是否为随机的
clusteringCoeRandom_strength = clustering_random(GraphOutDegree)
clusteringCoeRandom_degree = clustering_random(degree_sequence)
# transitivity
transitivity = nx.transitivity(G)
# 绘制clustering coefficient 参数分布
cc_sequence = np.array(list(clusteringCoeWeighted.values()))
counts, bins_edges = scaleFreeDraw(cc_sequence, 40, '无权')
k, probK = scaleFreeLogDraw(cc_sequence, 40)
fittingEx(k[4:], probK[4:])
fittinglinear(k[4:], probK[4:])


# 计算Katz中心性，计算out-edges Katz 中心性，所以为了计算G的out-edges，使用networkx中的katz函数，需要先反转G
# 计算结果不顺利， 需要绘图检查
Gv = G.reverse()
# 不太好用
katzCentralityWeighted = nx.katz_centrality(Gv, alpha=0.1, beta=1e-6, max_iter=int(1e4), tol=1e-6,
                                            nstart=None, normalized=True, weight='weight')
draw_properties(G, katzCentralityWeighted, factor=2000, title_name='Katz Centrality Weighted', region=region)
# katzCentralityUnWeighted = nx.katz_centrality(Gv, alpha=0.1, beta=0.0001, max_iter=int(1e3), tol=1e-6,
#                                             nstart=None, normalized=True)
# 计算eigenvectorCentrality
eigenCentralityWeighted = nx.eigenvector_centrality(Gv, max_iter=1000, tol=1e-6, nstart=None, weight='weight')
draw_properties(G2, eigenCentralityWeighted, factor=500, title_name='Eigen Centrality Weighted', region=region)
eigenCentralityUnweighted = nx.eigenvector_centrality(Gv, max_iter=1000, tol=1e-6, nstart=None, weight=None)
draw_properties(G2, eigenCentralityUnweighted, factor=2000, title_name='Eigen Centrality UnWeighted', region=region)

# 计算度中心性
degreeCentrality = nx.degree_centrality(G)
draw_properties(G2, degreeCentrality, factor=2000, title_name='degree Centrality', region=region)

# 计算reciprocity
# 重新回到节点的特性
# The reciprocity of a directed graph is defined as the ratio of the number of edges pointing in both
# directions to the total number of edges in the graph. Formally,
reciprocity = nx.reciprocity(G)

# 计算betweenness centrality
betweennessCentralityWeighted = nx.betweenness_centrality(G, weight='weight', seed=1)
draw_properties(G2, betweennessCentralityWeighted, factor=2000, title_name='betweenness Centrality Weighted', region=region)
betweennessCentralityUnweighted = nx.betweenness_centrality(G, weight=None, seed=1)
draw_properties(G2, betweennessCentralityUnweighted, factor=2000, title_name='betweenness Centrality UnWeighted', region=region)



def calculate_omega(g, size_of_component=100, aim_pmatr='omega'):
    # 去除较小的组件，之后再计算omega
    gtemp = g.copy()
    gtemp = gtemp.to_undirected()  # 转化为无向图
    isoNode = list(nx.isolates(gtemp))
    gtemp.remove_nodes_from(isoNode)  # 去除孤立点
    graphs = list(nx.connected_components(gtemp))

    '''
    for comp in graphs:
        if len(comp) >= size_of_component:
            pmatr = nx.omega(gtemp.subgraph(comp), nrand=1, seed=1)
            print(pmatr)
            print(len(comp))
    '''
    for comp in graphs:
        if len(comp) <= size_of_component:
            gtemp.remove_nodes_from(comp)


    '''
    for i in range(1, len(graphs)):
        gtemp.remove_nodes_from(graphs[i])
    '''
    if aim_pmatr == 'omega':
        pmatr = nx.omega(gtemp, niter=4, nrand=1, seed=1)
    else:
        pmatr = nx.sigma(gtemp, niter=4, nrand=1, seed=1)
    # sigma = nx.omega(GUndirected, seed=1)  # sigma 为 C/Cr / L/Lr
    return pmatr, gtemp


OMEGA, _ = calculate_omega(G2)
# similarity

# assortative mixing
rWeighted = nx.degree_pearson_correlation_coefficient(G, weight='weight')
rUnweighted = nx.degree_pearson_correlation_coefficient(G, weight=None)
rWeighted2 = nx.degree_pearson_correlation_coefficient(G2, weight='weight')
rUnweighted2 = nx.degree_pearson_correlation_coefficient(G2, weight=None)



# 分析网络的assortative mixing, 一个节点所连接的节点的度
k2k = np.zeros((len(G3.nodes), 2))
for node, i in zip(G3.nodes, range(len(k2k))):
    k2k[i, 0] = G3.degree(node)
    neighbors = list(G3.neighbors(node))
    if len(neighbors) != 0:
        mean_k = sum(sorted([d for n, d in G3.degree(neighbors)])) / len(neighbors)
        k2k[i, 1] = mean_k
plt.rcParams['figure.figsize'] = (6, 6)
plt.scatter(k2k[:, 0], k2k[:, 1])
plt.show()

num, edges = np.histogram(k2k[:, 0], bins=10)
nn = k2k[:, 0]
kk = k2k[:, 1]
amk = np.zeros(len(num))
for i in range(len(num)):
    if num[i] != 0:
        amk[i] = np.sum(kk[np.logical_and(nn > edges[i], nn <= edges[i+1])]) / num[i]
    else:
        amk[i] = np.inf
plt.scatter(edges[:-1], amk)
plt.show()

# average_neighbor_degree
# ang = nx.average_neighbor_degree(G, weight=None, source='in+out', target='in+out')
ang = nx.average_neighbor_degree(G, weight=None, source='in+out', target='in+out')
ang_list = np.array(list(ang.values()))
node_degree = np.array([d for _, d in G.degree(ang.keys())])
k2k = np.vstack((node_degree, ang_list)).T
k2k_dataframe = pd.DataFrame(k2k, columns=['k', 'k_avg'])
r_spearman = sta.spearmanr(node_degree, ang_list)
sns.jointplot(k2k_dataframe, x='k', y='k_avg')
plt.text(70, 10, s=rf'r' + r'$_{spearman}$ : ' +
                   f'{r_spearman[0]:.3}\np-value:{r_spearman[1]:.2e}', fontfamily='Times New Roman')
plt.show()



counts, bins_edges = scaleFreeDraw(ang_list, 30)
k, probK = scaleFreeLogDraw(ang_list, 30)
fittingEx(k, probK)
fittinglinear(k[3:], probK[3:])

# hierarchical characteristics
hierarchicalCoeUnWeighted = nx.flow_hierarchy(G, weight=None)
hierarchicalCoeWeighted = nx.flow_hierarchy(G, 'weight')


# 将想要的点输出给gmt绘图
gmt_path = 'GMTplot/'
pos = nx.get_node_attributes(G, 'position')
size_fac = 10  # 调整大小
min_size = 0.005  # 点的最小大小
bas_fac = 1000  # 对数函数的底

pos_arr = dict2array(pos)
prop_arr = dict2array(degreeCentrality)
print("最小值为{}，最大值为{}".format(np.min(prop_arr), np.max(prop_arr)))

size_arr = max_min_scale(prop_arr, 1, 0)
size_arr[size_arr < min_size] = min_size

size_arr = np.log(size_arr + 1) / np.log(bas_fac)
all_arr = np.hstack((pos_arr, prop_arr))
all_arr = np.hstack((all_arr, size_arr * size_fac))

pd_arr = pd.DataFrame(all_arr)
pd_arr.to_csv(gmt_path + region + '_points.dat', sep='\t', header=None, index=None)


# 下面绘制线段
edges = G.edges(data=True)
vectors = np.zeros((len(edges), 4), np.float64)
k = 0
for i, j, data in edges:

    # 计算边两点之间的距离与角度
    pos1 = pos[i]
    pos2 = pos[j]
    ll = Geodesic.WGS84.Inverse(pos1[1], pos1[0], pos2[1], pos2[0])
    vectors[k, 0] = pos1[0]
    vectors[k, 1] = pos1[1]
    vectors[k, 2] = ll['azi1']
    vectors[k, 3] = ll['s12'] / 1000
    k = k + 1
vectors_pd = pd.DataFrame(vectors)
vectors_pd.to_csv(gmt_path + region + '_vector.dat', sep='\t', header=None, index=None)


# !!3 新的讨论与分析
# node out strength 与节点的地震数量与震级，最大震级？
nodes_with_info = []
events_num = []
max_magnitude = []
for node_id in G.nodes:
    event_id = np.array(np.where(event_id2node_id == node_id)).reshape(-1)
    event_info = catalog[event_id - 1]  # 读取每个节点的地震的地震目录
    max_m = np.max(event_info[:, 3])  # 节点最大震级
    nodes_with_info.append(node_id)
    num = np.sum(event_id2node_id == node_id)
    max_magnitude.append(max_m)
    events_num.append(num)
outdegree = [d for ii, d in G.out_degree(weight='weight')]  # out node strength
indegree = [d for ii, d in G.in_degree(weight='weight')]  # out node strength
Rspear_num_outdegree = sta.spearmanr(events_num, outdegree)
Rspear_outdegree_maxM = sta.spearmanr(outdegree, max_magnitude)
Rspear_num_maxM = sta.spearmanr(events_num, max_magnitude)
plt.scatter(outdegree, events_num)
plt.show()

# 通过hits 找到hub and authority
(hubs, authorities) = nx.hits(G2)

hubs_arr = dict2array(hubs)
hubs_log = np.log10(np.array(hubs_arr))

draw_properties(G2, list(hubs_log), region=region, factor=5e5)
xx, yy, _, _ = scaleFreeLogPinDraw(hubs_log, bins=30)
'''
# logarithmic binning ???!!!
# 绘制logarithmic binning
ksLog = np.log10(GraphWeight[GraphWeight != 0])
bins = 10
xMax = np.max(ksLog)
xMin = np.min(ksLog)
xStart = np.log(xMin)
xEnd = np.log(xMax)
logEdges = np.linspace(xStart, xEnd, num=bins + 1, endpoint=True)
lineEdges = np.power(np.e, logEdges)


# 对数轴案例
# histogram on log scale.
# Use non-equal bin sizes, such that they look equal on log scale.
x = [2, 1, 76, 140, 286, 267, 60, 271, 5, 13, 9, 76, 77, 6, 2, 27, 22, 1, 12, 7,
     19, 81, 11, 173, 13, 7, 16, 19, 23, 197, 167, 1]
x = pd.Series(x)

# histogram on linear scale
plt.subplot(211)
hist, bins, _ = plt.hist(x, bins=8)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(x, bins=logbins)
plt.xscale('log')
plt.yscale('log')
plt.show()



# 关于assortative mixing的论述
Thus, a possible explanation of the pattern of r-values seen in Table 10.1
is that most networks are naturally disassortative by degree because they are
simple networks, while social networks (and perhaps a few others) override
this natural bias and become assortative by virtue of their group structure.

# 自己写的out degree 已经用不上了(lll￢ω￢)
GraphWeight = np.zeros(len(G.nodes))
GraphNo = G.nodes
i = 0
for node in GraphNo:
    for link in G.edges(node):
        GraphWeight[i] = GraphWeight[i] + G[link[0]][link[1]]['weight']
    i = i + 1
'''