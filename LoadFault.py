import numpy as np
import geotable
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
# input filename
# output structure faults
# faults info & faults points

# LoadFault 为加载GMT CN-faults断层文件


class FAULT:
    # 定义断层类，有原始的断层文本信息，和断层经纬点
    # border方法依照断层经纬点，输出断层的经纬范围
    def __init__(self, info=None, points=None):
        self.info = info
        self.points = points

    def border(self):
        if self.points.size <= 0:
            print("This fault is empty.")
            return "This fault is empty."
        else:
            min_long = np.min(self.points[:, 0])
            max_long = np.max(self.points[:, 0])
            min_lat = np.min(self.points[:, 1])
            max_lat = np.max(self.points[:, 1])
            return np.array([min_long, max_long, min_lat, max_lat])


def gmtloadfault(faultDataName = 'data/CN-faults.txt'):
    # 将GMT的断层文件读取
    # 将所有断层放入faultList变量，每个断层用FAULT类别
    faultList = []
    faultInfo = "None"
    faultPoints = []
    f = open(faultDataName)
    line = f.readline()
    findChar = "|"  # 用于后面的find函数，以判断断层是否有基本信息
    i = 0  # 给断层标号
    while line:
        if line == ">\n":
            faultPoints = np.array(faultPoints, np.float64)
            if i != 0:
                exec("FAULT%d = FAULT(faultInfo, faultPoints)" % i)
                exec("faultList.append(FAULT%d)" % i)
            faultPoints = []
            i = i+1
            line = f.readline()
            if line.find(findChar) != -1:  # 如果存在findChar 则该行为断层基本信息
                faultInfo = line
                line = f.readline()
            else:
                faultInfo = "None"
        else:
            point = line.split()
            # fault_lines = LineString(point)
            faultPoints.append(point)
            line = f.readline()
    # 结尾的最后一个断层
    faultPoints = np.array(faultPoints, np.float64)
    exec("FAULT%d = FAULT(faultInfo, faultPoints)" % i)
    f.close()
    return faultList


def fault_into_line(fault_list):
    # 因为数据格式不一样，将fault类型改变了shapely的line类型
    fault_line_list = []
    for fault in fault_list:
        fault = LineString(fault.points)
        fault_line_list.append(fault)
    return fault_line_list

def kml_load_fault(filename="data/GEM/kml/gem_active_faults.kml"):
    faults_list = geotable.load(filename)
    faults_line = faults_list.geometries
    return faults_line


def kmzloadfault(filename='qfaults.kmz', region='southern_california'):
    # 读取USGS_kmz文件中的断层数据
    faults = geotable.load('data/' + region + '/' + filename)
    faults_historical = faults[faults['geometry_layer'] == 'Historical (150 years)']
    # faults 拥有 'Name' 'geometry_object'  'geometry_layer'  'geometry_proj4'四个属性
    # geometry_layer
    # 'Historical (150 years)', 'Late Quaternary (130,000 years)', 'Latest Quaternary (15,000 years)',
    # 'Class B (various ages)', 'Middle and Late Quaternary (750,000 years)', 'Unspecified Age',
    # 'Undifferentiated Quaternary (1.6 millions years)'
    fault_info = "None"
    faults_list = []
    for record in faults_historical['geometry_object']:
        for line in record.geoms:
            fault_temp = FAULT(info=None, points=line)
            faults_list.append(fault_temp)
    return faults_list


# usage
# faultName = 'data/CN-faults.txt'
# faultList = gmtloadfaults()

if __name__ == '__main__':

    faults_list = kml_load_fault()
    line_list = faults_list.geometries
    border_polygon = Polygon([(5, 35), (20, 35), (20, 48), (5, 48)])
    for faults in line_list:
        if border_polygon.contains(faults):
            x = faults.xy[0]
            y = faults.xy[1]
            plt.plot(x, y)
    plt.show()
