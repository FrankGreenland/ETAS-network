from locale import str
import numpy as np
import pandas as pd

from LoadFault import gmtloadfault, fault_into_line, kmzloadfault, kml_load_fault
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from geopy import distance
import networkx as nx
from LoadEarthquakes import load_pmatr, load_etas
from scipy.spatial import KDTree
import scipy.stats as sta

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['figure.figsize'] = (8, 14)  # shanxi
# plt.rcParams['figure.figsize'] = (15, 11)  # southern california

plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
# nLen = 5  # 单位km
# thresholdProb = 1e-2  # 最小的关系
# region = 'southern california'
# 'shanxi' 'southern california'

# 通过box counting 计算断层的分形维数，并生成不同尺度下的Node


def get_border_file(region='shanxi'):
    # 通过范围文件 获得区域范围
    f = open("data/" + region + "/border.txt")
    border = f.readline()
    border = np.array(border.split(), np.float64)
    f.close()
    return border


def select_fault(borders, fault_list, isLine=True):
    # 输入断层，输入研究区域，输出在研究范围内的断层
    fault_list_in = []
    polygon = Polygon([(borders[0], borders[2]),  # 由研究区域生成图形类
                       (borders[1], borders[2]),
                       (borders[1], borders[3]),
                       (borders[0], borders[3])
                       ])
    x, y = polygon.exterior.xy  # 矩形的外沿
    plt.plot(x, y)
    if isLine:
        for fault in fault_list:
            if polygon.intersects(fault):
                fault_list_in.append(fault)
                x, y = fault.xy
                plt.plot(x, y)
    else:
        for fault in fault_list:
            line = LineString(fault.points)  # 每个断层生成一个线类
            if polygon.intersects(line):  # 判断线与矩形是否在一块
                fault_list_in.append(fault)
                x, y = line.xy
                plt.plot(x, y)

    plt.show()
    return fault_list_in


def border_length(border):
    # 研究区域的大致经度长度跨度， 纬度的长度跨度
    loc1 = (border[2], border[0])
    loc2 = (border[3], border[0])
    lat_distance = distance.geodesic(loc1, loc2).km

    loc1 = (0.5*border[2] + 0.5*border[3], border[0])
    loc2 = (0.5*border[2] + 0.5*border[3], border[1])
    long_distance = distance.geodesic(loc1, loc2).km
    print("纬度长度差为" + str(lat_distance) +"km\n经度长度差约为" + str(long_distance) + "km\n")
    return long_distance, lat_distance


def exchange_coordinate2distance(pll, border, latdistance, longdistance):
    # 由点的经纬度换算到欧式地面位置
    pll = np.array(pll)
    sll = np.array((border[0], border[2]))
    ell = np.array((border[1], border[3]))
    # 将目标区域粗算为矩形后，将该区域的经纬度坐标，改为位置坐标
    alldistance = np.array((latdistance, longdistance))
    aimdistance = alldistance * (pll - sll) / (ell - sll)
    return aimdistance


def exchange_distance2coordinate(dp, latd, longd, border):
    # 将距离转换为经纬度
    # dp为欧式位置
    dp = np.array(dp)
    d_distance = np.array((latd, longd))
    o_coord = np.array((border[0], border[2]))
    end_coord = np.array((border[1], border[3]))
    cp = o_coord + (end_coord - o_coord) * dp / d_distance
    return cp


def box_counting(G, border, latDistance, longDistance, faultinregion, nlen=5):
    faultInRegion = faultinregion
    f_empty = open('GMT_boxes/boxes_empty.dat', 'w')
    f_full = open('GMT_boxes/boxes_full.dat', 'w')
    # 输入边界，边界长度，格子长度 （单位km）
    # 将有断层的节点设为Node， 并返回box的数量
    regionBorder = border
    # 下面是一些算格子的问题
    yN = int(np.around(latDistance / nlen))  # 经纬度方向大约有多少个格子
    xN = int(np.around(longDistance / nlen))
    # 将一个格子的大小转化为经纬度
    nyLen = (regionBorder[3] - regionBorder[2]) * nlen / latDistance  # 一个格子的经纬度跨度
    nxLen = (regionBorder[1] - regionBorder[0]) * nlen / longDistance

    originalPosition = (regionBorder[0], regionBorder[2])  # 起始点

    yEndPoint = (yN - 1) * nyLen + regionBorder[2]  # 结束的点位
    xEndPoint = (xN - 1) * nxLen + regionBorder[0]

    xPositions = np.linspace(originalPosition[0], xEndPoint, xN)  # 生成每个格子左下角的点
    yPositions = np.linspace(originalPosition[1], yEndPoint, yN)

    all_boxs = []
    boxs = []
    boxes_coord = []
    bN = 0
    # fig, ax = plt.subplots(1, 1)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    for fault in faultInRegion:
        line = fault
        x, y = line.xy
        # plt.plot(x, y)
    for i in xPositions:
        for j in yPositions:
            box = Polygon([
                (i, j),
                (i + nxLen, j),
                (i + nxLen, j + nyLen),
                (i, j + nyLen)
            ])
            x, y = box.exterior.xy  # 矩形的外沿
            # bb = plt.Polygon(xy=[[i, j], [i + nxLen, j], [i + nxLen, j + nyLen], [i, j + nyLen]], color='black',
            #                  alpha=0.2)
            f_empty.write(f"{i} {j}\n"
                          f"{i+nxLen} {j}\n"
                          f"{i+nxLen} {j+nyLen}\n"
                          f"{i} {j+nyLen}\n>\n")
            all_boxs.append([i + 0.5*nxLen, j + 0.5*nyLen])
            for fault in faultInRegion:
                line = fault
                if box.intersects(line):
                    # bb = plt.Polygon(xy=[[i, j], [i + nxLen, j], [i + nxLen, j + nyLen], [i, j + nyLen]], color='red',
                    #                  alpha=0.3)
                    # ax.add_patch(bb)
                    f_full.write(f"{i} {j}\n"
                                 f"{i+nxLen} {j}\n"
                                 f"{i+nxLen} {j+nyLen}\n"
                                 f"{i} {j+nyLen}\n>\n")
                    boxs.append(
                        exchange_coordinate2distance((i + 0.5*nxLen, j + 0.5*nyLen), border, latDistance, longDistance)
                    )
                    boxes_coord.append([i + 0.5*nxLen, j + 0.5*nyLen])
                    G.add_node(bN, position=(i + 0.5*nxLen, j + 0.5*nyLen), num_earthquake=0)  # , fault=fault.info
                    bN = bN + 1
                    break
            # ax.add_patch(bb)
    f_empty.close()
    f_full.close()
    # plt.title("边长为" + str(nlen) + "km")
    # plt.show()
    # plt.savefig("figures/caogao/" + str(nlen) + '.jpg')
    # boxes为欧式下的坐标，boxes_coord为经纬度下盒子的中心，all_boxes为所有的盒子，包括不与断层相交的盒子
    return bN, boxs, boxes_coord, all_boxs


def slope(G, regionBorder, latDistance, longDistance, faultInRegion, eE=np.array([2, 5, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60])):
    # 下面算该区域断层的box dimension
    eN = np.zeros(eE.shape)
    for i in range(0, len(eE)):
        print(i)
        eN[i], _ = box_counting(G, regionBorder, latDistance, longDistance, faultinregion=faultInRegion, nlen=eE[i])
    plt.clf()
    plt.cla()
    # 计算回归系数 简单线性拟合
    logeE = np.log(1 / eE)
    logeN = np.log(eN)
    slope, intercept = np.polyfit(logeE, logeN, 1)
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.scatter(logeE, logeN)
    plt.plot(logeE, slope * logeE + intercept, color='red')
    plt.xlabel('log(1/e)')
    plt.ylabel('logN')
    plt.title('slope:' + str(slope))
    plt.show()
    plt.savefig('figures/caogao/slope:' + str(slope) + '.jpg')
    return


def boxes_drew(boxes, border, longd, latd, nlen, filename='gmt_boxes/boxes.dat'):
    # 输出boxes的gmt绘制文件
    # boxes为之前计算出的盒子，border为研究区域的经纬范围，longd为经度的长度，latd为纬度的长度，filename为写在什么地方
    # 未完成
    f = open(filename, 'w')
    nyLen = (border[3] - border[2]) * nlen / latd  # 一个格子的经纬度跨度
    nxLen = (border[1] - border[0]) * nlen / longd
    for box in boxes:
        f.write()
    f.close()
    return 1


def graphGenerate(region, nlen=32, thresholdprob=0.01, thdistance=5, threshold_m=1.5):
    G = nx.DiGraph()
    thresholdProb =thresholdprob
    nLen = nlen  # 一个格子的大小，单位为km
    regionBorder = get_border_file(region)  # 获取研究区域的边界值
    if region == 'shanxi':
        faultsList = gmtloadfault()  # 读取GMT中的中国断层
        faultsList = fault_into_line(faultsList)
        plt.rcParams['figure.figsize'] = (8, 14)  # shanxi
        isline = True
    elif region == 'INGV' or region == 'southern_california':
        faultsList = kml_load_fault()  # 读取GEM数据库中的断层
        if region ==' INGV':
            plt.rcParams['figure.figsize'] = (12, 14)  # Italy
        else:
            plt.rcParams['figure.figsize'] = (15, 11)  # southern california
        isline = True
    elif region == 'chuandian':
        faultsList = gmtloadfault()  # 读取GMT中的中国断层
        faultsList = fault_into_line(faultsList)
        plt.rcParams['figure.figsize'] = (8, 12)  # chuandian
        isline = True
    else:
        print('region is not studied')
        # faultsList = kmzloadfault()  # 读取南加州
    faultInRegion = select_fault(regionBorder, faultsList, isLine=isline)  # 找到研究区域的断层
    longDistance, latDistance = border_length(regionBorder)  # 计算研究区域长度跨度
    # eE = np.logspace(0, 6, num=10, base=2)
    # slope(G, regionBorder, latDistance, longDistance, faultInRegion, eE)
    # eE=np.linspace(1,100,50 -4 +1)
    boxN, boxes, boxes_coord, all_boxes = box_counting(G, regionBorder, latDistance, longDistance, faultInRegion, nLen)  # boxcounting划分格子，并找到节点
    nx.get_node_attributes(G, "position")  # 测试图的属性
    catalog = load_etas(region, threshold_m)  # 加载etas地震数据
    pmatr = load_pmatr(region, thresholdProb)  # 加载地震数据的概率关系
    catalogArray = np.array(catalog)
    # 将地震的经纬度转化为位置
    earthquakePosition = exchange_coordinate2distance(catalogArray[:, 1:3], regionBorder, latDistance, longDistance)

    ## 需要设置一个阈值，地震和断层太远也不行
    # 将box为KDTree，方便找到最近邻
    kdTree1 = KDTree(np.array(boxes))
    # kdTree2 = KDTree(earthquakePosition)
    # 计算最近邻
    dd, ii = kdTree1.query(earthquakePosition, k=1)  # dd为最近的距离，ii为对应的box号
    boxWithEarthquake = len(set(ii))  # 有数据的节点
    pmatrEarthquake = pmatr[:, 0:2].astype(int) - 1   # pmatrEarthquake为地震号
    node2node = ii[pmatrEarthquake]
    node2node_distance = dd[pmatrEarthquake]
    SelfRatio = 1 - sum((node2node[:, 1] - node2node[:, 0]).astype(bool)) / len(node2node)
    print("自引率为" + str(SelfRatio))
    print("节点数为" + str(boxWithEarthquake))
    # 给节点加地震数量
    for i in range(0, len(ii)):
        # if dd[i] <= thdistance:  # 判断如果距离太远了，这个地震就不要了
            G.nodes[ii[i]]['num_earthquake'] = G.nodes[ii[i]]['num_earthquake'] + 1

    # 除掉起始的N1
    # 建立图的边
    for i in range(0, len(node2node)):
        # flag = node2node_distance[i] <= thdistance
        # if flag.all():  # 判断如果距离太远了，这个地震就不要了
        G.add_edge(node2node[i, 0], node2node[i, 1], weight=0)
        G[node2node[i, 0]][node2node[i, 1]]['weight'] = G[node2node[i, 0]][node2node[i, 1]]['weight'] + pmatr[i, 2]  # / G.nodes[node2node[i, 0]]['num_earthquake']
        # 这里可以看下除不除那个分母的效果
    print("`    ")
    # 去除孤立点
    isoNode = list(nx.isolates(G))
    G.remove_nodes_from(isoNode)
    event_id2node_id = ii
    event_id = np.linspace(1, len(catalogArray), len(catalogArray))
    catalogArray[:, 0] = event_id
    # 输出依次为生成的图，图对应的boxes坐标，地震事件与nodeid的对应，etas模型获得的pmatr, 带id的地震目录
    return G, boxes, event_id2node_id, pmatr, catalogArray, boxes_coord, all_boxes


def output_position(catalog_pd, size_f, fname):
    arr = np.array(catalog_pd)[:, 1:4]
    arr[:, 2] = arr[:, 2] * size_f
    pd.DataFrame(arr).to_csv('./GMTplot/position/' + fname, sep='\t', header=None, index=None)
    return 1


def earthquake2node(event_id):
    # 给地震的序号，返回其对应的节点
    return 0


def child_event(event_id, pmatr, thp):
    # 给定一个地震事件，找到其子事件，由pmatr做限定
    pmatr_temp = pmatr[pmatr[:, 2] >= thp]
    offspring = np.array([])
    event_id = np.array(event_id, int).reshape(-1, 1)
    for id in event_id:
        child_id = pmatr_temp[pmatr_temp[:, 0] == id]
        offspring = np.hstack((offspring, child_id[:, 1]))
    return offspring


def whole_offspring(event_id, pmatr, thp):
    pmatr_temp = pmatr[pmatr[:, 2] >= thp]
    offsprings = np.array([])
    offspring = np.array(event_id).reshape(-1)
    while offspring.size > 0:
        offsprings = np.hstack((offsprings, offspring))
        next_offspring = child_event(offspring, pmatr_temp, thp)
        offspring = np.array(list(set(next_offspring)), int)

    return np.array(list(set(offsprings)), int)

def drew_box(all_boxes, boxes_coord, nxLen, nyLen, fpath='plot_boxes_info/', info=0, region='shanxi'):
    f_empty = open(fpath + 'empty_boxes.dat', 'w')
    f_full = open(fpath + 'boxes_info.dat', 'w')
    for i, j in all_boxes:
        f_empty.write(f"{i} {j}\n"
                      f"{i + nxLen} {j}\n"
                      f"{i + nxLen} {j + nyLen}\n"
                      f"{i} {j + nyLen}\n>\n")
    for i, j in boxes_coord:
        f_full.write(f"{i} {j}\n"
                     f"{i+nxLen} {j}\n"
                     f"{i+nxLen} {j+nyLen}\n"
                     f"{i} {j+nyLen}\n>\n")
    f_empty.close()
    f_full.close()
    return 1


def drew_community(all_boxes, community_list, boxes_arr, nxLen, nyLen, latDistance, longDistance, regionBorder, fpath='GMT_plot_community_combine/', info=0, region='shanxi'):
    num_comm = len(community_list)
    boxes = np.array(boxes_arr)
    f_empty = open(fpath + 'empty_boxes.dat', 'w')
    f_full = open(fpath + 'boxes_info.dat', 'w')
    k = 0
    for comm in community_list:
        comm_arr = np.array(list(comm))
        boxes_pos = boxes[comm_arr]
        boxes_coord = exchange_distance2coordinate(boxes_pos, latDistance, longDistance, regionBorder)
        for i, j in boxes_coord:
            f_full.write(f">-Z{k}\n")
            f_full.write(f"{i} {j}\n"
                         f"{i + nxLen} {j}\n"
                         f"{i + nxLen} {j + nyLen}\n"
                         f"{i} {j + nyLen}\n")
        k = k + 1

    for i, j in all_boxes:
        f_empty.write(f"{i} {j}\n"
                      f"{i + nxLen} {j}\n"
                      f"{i + nxLen} {j + nyLen}\n"
                      f"{i} {j + nyLen}\n>\n")
    print(f"共有{k}个communities")
    f_empty.close()
    f_full.close()
    return 1


def drew_community_region(region, all_boxes, community_list, boxes_arr, nxLen, nyLen, latDistance, longDistance, regionBorder, fpath='GMT_plot_community/', info=0):
    num_comm = len(community_list)
    boxes = np.array(boxes_arr)
    f_empty = open(fpath + f'{region}_empty_boxes.dat', 'w')
    f_full = open(fpath + f'{region}_boxes_info.dat', 'w')
    k = 0
    for comm in community_list:
        comm_arr = np.array(list(comm))
        boxes_pos = boxes[comm_arr]
        boxes_coord = exchange_distance2coordinate(boxes_pos, latDistance, longDistance, regionBorder)
        for i, j in boxes_coord:
            f_full.write(f">-Z{k}\n")
            f_full.write(f"{i} {j}\n"
                         f"{i + nxLen} {j}\n"
                         f"{i + nxLen} {j + nyLen}\n"
                         f"{i} {j + nyLen}\n")
        k = k + 1

    for i, j in all_boxes:
        f_empty.write(f"{i} {j}\n"
                      f"{i + nxLen} {j}\n"
                      f"{i + nxLen} {j + nyLen}\n"
                      f"{i} {j + nyLen}\n>\n")
    print(f"共有{k}个communities")
    f_empty.close()
    f_full.close()
    return 1

if __name__ == '__main__':
    # 测试这部分的函数
    region = 'shanxi'
    nLen = 16
    mProb = 1e-1
    regionBorder = get_border_file(region)  # 获取研究区域的边界值
    longDistance, latDistance = border_length(regionBorder)  # 计算研究区域长度跨度
    thDistance = 5 + nLen / 2  # 默认为5，如果想去掉这个限制，可以输入一个很大的数，比如10000
    thm = 2.4  # shanxi thm=2.5 |||| INGV  thm=2.9|||| Chuandian  thm=2.8||| southern california thm=2.4  很重要，不然会有bug
    G, boxes, event_id2node_id, pmatr, catalog, boxes_coord, all_box = graphGenerate(region, nLen, mProb, thDistance, thm)
    boxes_arr = np.array(boxes)
    nyLen = (regionBorder[3] - regionBorder[2]) * nLen / latDistance  # 一个格子的经纬度跨度
    nxLen = (regionBorder[1] - regionBorder[0]) * nLen / longDistance
    ###

    mag = 6.0
    flag = np.where(catalog[:, 3] >= mag)
    catalog_temp = catalog[flag]
    thp = 0.5
    temp_id = catalog_temp[:, 0]
    offspring = child_event(temp_id, pmatr, thp)  # 由地震号找到其子代
    offspring_all = whole_offspring(temp_id, pmatr, thp)  # 包含子代的子代
    boxes_with_offspring = event_id2node_id[offspring_all.astype(int)-1]
    boxes_position = boxes_arr[boxes_with_offspring]
    boxes_position_coord = exchange_distance2coordinate(boxes_position, latDistance, longDistance, regionBorder)  # 将boxes转化为经纬度
    drew_box(all_box, boxes_position_coord, nxLen, nyLen)



    box_coord = exchange_distance2coordinate(boxes, latDistance, longDistance, regionBorder)  # 将boxes转化为经纬度
    plt.scatter(box_coord[:, 0], box_coord[:, 1], c='blue')
    plt.scatter(catalog_temp[0, 1], catalog_temp[0, 2], c='red')
    aaa = exchange_distance2coordinate(boxes[350], latDistance, longDistance, regionBorder)
    plt.scatter(aaa[0], aaa[1], c='yellow')
    plt.show()

    # 演示这些在加一减一上怎么使用
    boxes_id = event_id2node_id[np.array(1426-1, int)]  # 由事件号找到box,node的id号
    boxes_id = event_id2node_id[np.array(temp_id-1, int)]
    events_info = catalog[1426 - 1]  # 由事件id查询事件信息
    events_info = catalog[temp_id - 1]
    boxes[boxes_id]  # 查询boxes的经纬度

    # 由node号找到其对应的事件的演示
    node_id = 21
    event_id = np.array(np.where(event_id2node_id == node_id)).reshape(-1)
    plt.scatter(box_coord[:, 0], box_coord[:, 1], c='blue')
    plt.scatter(box_coord[node_id, 0], box_coord[node_id, 1], c='pink')
    plt.scatter(catalog[event_id][:, 1], catalog[event_id][:, 2], c='red')
    plt.show()


    # 找到有信息的节点号
    nodes_with_info = []
    events_num = []
    for node_id in G.nodes:
        nodes_with_info.append(node_id)
        num = np.sum(event_id2node_id == node_id)
        events_num.append(num)
    outdegree = [d for ii, d in G.out_degree(weight='weight')]  # out node strength
    Rspear = sta.spearmanr(events_num, outdegree)
    plt.scatter(outdegree, events_num)
    plt.show()

# usage example
# G = graphGenerate(region, nLen, thresholdProb)
'''
# 绘制box counting 示意图
region = 'shanxi'
plt.rcParams['figure.figsize'] = (8, 14)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
border = get_border_file(region)
a, b = border_length(border)
slope(border, a, b)

plt.rcParams['figure.figsize'] = (8, 14)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
# 尝试画G图
pos1 = nx.get_node_attributes(G, 'position')
# drawDensity = 10
# for key in pos1.keys():
#    pos1[key] = (pos1[key][0] * drawDensity, pos1[key][1] * drawDensity)
nx.draw(G, pos1, node_size=60, alpha=0.8, with_labels=True, font_size=6, width=0.5)
plt.title('有向图')
plt.show()

# 去除自环的图
G2 = G.copy()
G2.remove_edges_from(nx.selfloop_edges(G2))
pos1 = nx.get_node_attributes(G2, 'position')
nx.draw(G2, pos1, node_size=60, alpha=0.6, with_labels=True, font_size=6, width=0.5)
plt.title('有向图去除自环')
plt.show()


检测最近的断层和节点对不对
boxs_array = np.array(boxes)
plt.scatter(boxs_array[:, 0], boxs_array[:, 1], s=10)
kk = 2121
plt.scatter(boxs_array[ii[kk], 0], boxs_array[ii[kk], 1], c='red', s=8)
plt.scatter(earthquakePosition[kk, 0], earthquakePosition[kk, 1], c='green', s=8)
plt.scatter(earthquakePosition[:, 0], earthquakePosition[:, 1], s=1, color='grey')
plt.show()

def graphGenerate(region, nlen=32, thresholdprob=0.01, thdistance=5):
    G = nx.DiGraph()
    thresholdProb =thresholdprob
    nLen = nlen  # 一个格子的大小，单位为km
    regionBorder = get_border_file(region)  # 获取研究区域的边界值
    if region == 'shanxi':
        faultsList = gmtloadfault()  # 读取GMT中的中国断层
        faultsList = fault_into_line(faultsList)
        plt.rcParams['figure.figsize'] = (8, 14)  # shanxi
        isline = True
    elif region == 'INGV' or region == 'southern_california':
        faultsList = kml_load_fault()  # 读取GEM数据库中的断层
        if region ==' INGV':
            plt.rcParams['figure.figsize'] = (12, 14)  # Italy
        else:
            plt.rcParams['figure.figsize'] = (15, 11)  # southern california
        isline = True
    elif region == 'chuandian':
        faultsList = gmtloadfault()  # 读取GMT中的中国断层
        faultsList = fault_into_line(faultsList)
        plt.rcParams['figure.figsize'] = (8, 12)  # chuandian
        isline = True
    else:
        print('region is not studied')
        # faultsList = kmzloadfault()  # 读取南加州
    faultInRegion = select_fault(regionBorder, faultsList, isLine=isline)  # 找到研究区域的断层
    longDistance, latDistance = border_length(regionBorder)  # 计算研究区域长度跨度
    # eE = np.logspace(0, 6, num=10, base=2)
    # slope(G, regionBorder, latDistance, longDistance, faultInRegion, eE)
    # eE=np.linspace(1,100,50 -4 +1)
    boxN, boxes = box_counting(G, regionBorder, latDistance, longDistance, faultInRegion, nLen)  # boxcounting划分格子，并找到节点
    nx.get_node_attributes(G, "position")  # 测试图的属性
    catalog = load_etas(region, 1.5)  # 加载etas地震数据
    pmatr = load_pmatr(region, thresholdProb)  # 加载地震数据的概率关系
    catalogArray = np.array(catalog)
    # 将地震的经纬度转化为位置
    earthquakePosition = exchange_coordinate2distance(catalogArray[:, 1:3], regionBorder, latDistance, longDistance)

    ## 需要设置一个阈值，地震和断层太远也不行
    # 将box为KDTree，方便找到最近邻
    kdTree1 = KDTree(np.array(boxes))
    # kdTree2 = KDTree(earthquakePosition)
    # 计算最近邻
    dd, ii = kdTree1.query(earthquakePosition, k=1)  # dd为最近的距离，ii为对应的box号
    boxWithEarthquake = len(set(ii))  # 有数据的节点
    pmatrEarthquake = pmatr[:, 0:2].astype(int) - 1   # pmatrEarthquake为地震号
    node2node = ii[pmatrEarthquake]
    node2node_distance = dd[pmatrEarthquake]
    SelfRatio = 1 - sum((node2node[:, 1] - node2node[:, 0]).astype(bool)) / len(node2node)
    print("自引率为" + str(SelfRatio))
    print("节点数为" + str(boxWithEarthquake))
    # 给节点加地震数量
    for i in range(0, len(ii)):
        # if dd[i] <= thdistance:  # 判断如果距离太远了，这个地震就不要了
            G.nodes[ii[i]]['num_earthquake'] = G.nodes[ii[i]]['num_earthquake'] + 1

    # 除掉起始的N1
    # 建立图的边
    for i in range(0, len(node2node)):
        # flag = node2node_distance[i] <= thdistance
        # if flag.all():  # 判断如果距离太远了，这个地震就不要了
        G.add_edge(node2node[i, 0], node2node[i, 1], weight=0)
        G[node2node[i, 0]][node2node[i, 1]]['weight'] = G[node2node[i, 0]][node2node[i, 1]]['weight'] + pmatr[i, 2]  # / G.nodes[node2node[i, 0]]['num_earthquake']
        # 这里可以看下除不除那个分母的效果
    print("`    ")
    # 去除孤立点
    isoNode = list(nx.isolates(G))
    G.remove_nodes_from(isoNode)
    return G


'''
