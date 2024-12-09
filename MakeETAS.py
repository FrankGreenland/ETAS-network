import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 生成给ETASR用的地震目录
threshold_mag = 2.5
# 读取数据
region = 'southern_california'
path = 'data/' + region + '/catalogs'
files = os.listdir(path)
earthquake_catalog = np.array(pd.read_csv(path + '/' + files[0], sep='\\s+'))
for i in range(1, len(files)):
    temp_catalog = np.array(pd.read_csv(path + '/' + files[i], sep='\\s+'))
    earthquake_catalog = np.vstack((earthquake_catalog, temp_catalog))



# 画下地震点位
plt.scatter(earthquake_catalog[:, 7], earthquake_catalog[:, 6])
plt.show()


# 生成给R使用的数据
earthquake_catalog_selected = earthquake_catalog[earthquake_catalog[:, 4] >= threshold_mag]
# 画下地震点位
plt.scatter(earthquake_catalog_selected[:, 7], earthquake_catalog_selected[:, 6])
plt.show()
catalog4R = np.hstack((earthquake_catalog_selected[:, 0:2], earthquake_catalog_selected[:, 7:8]))
catalog4R = np.hstack((catalog4R, earthquake_catalog_selected[:, 6:7]))
catalog4R = np.hstack((catalog4R, earthquake_catalog_selected[:, 4:5]))
pd.DataFrame(catalog4R).to_csv("data/southern_california/etasR.csv", sep=',', index=False,
                               header=['date', 'time', 'long', 'lat', 'mag'])


# 下面是linux etas make
def devide_split(array, sep='\\s+', aim_type=int):
    # 将numpy数组按照某个字符差分为多个数组
    array = array.astype(str)
    sarray = np.char.split(array, sep)
    length = len(sarray[0])
    num = len(sarray)
    darray = np.zeros((num, length), aim_type)
    for i in range(len(sarray)):
        for j in range(length):
            darray[i][j] = sarray[i][j]
    return darray


# 由ETASR的地震目录生成linux etas的地震目录
def R2linux_catalog(R_catalog_filename, thm=0, sep=','):
    catalog = pd.read_csv(R_catalog_filename, sep=sep)
    if thm:
        catalog = catalog[catalog['mag'] >= thm]
        print('共有地震数量：')
        print(len(catalog))
    year = np.array(catalog['date']).astype(str)
    darray = devide_split(year, sep='-', aim_type=int)
    # 这里两个目录的sep不一样 INGV是‘-’，南加州是‘/’
    year = darray[:, 0]
    month = darray[:, 1]
    day = darray[:, 2]
    long = np.array(catalog['long'])
    lat = np.array(catalog['lat'])
    mag = np.array(catalog['mag'])
    time = np.array(catalog['time'])
    darray = devide_split(time, sep=':', aim_type=float)
    hour = darray[:, 0].astype(int)
    minute = darray[:, 1].astype(int)

    return year, month, day, hour, minute, lat, long, mag


# 生成linux ETAS用的地震目录
def linux_etas_catalog(catalog_filename, region='shanxi', aim_filename='', thm=0, sep=','):
    # 生成一个区域的linux etas 地震目录，检测重复时间与重复位置

    catalog = np.array(pd.read_csv(catalog_filename))
    if thm:
        catalog = catalog[catalog[:, 4] >= thm]
    year, month, day, hour, minute, lat, long, mag = R2linux_catalog(catalog_filename, thm, sep=sep)
    syear = 2000
    catalog = catalog[year >= syear]
    month = month[year >= syear]
    day = day[year >= syear]
    hour = hour[year >= syear]
    minute = minute[year >= syear]
    lat = lat[year >= syear]
    long = long[year >= syear]
    mag = mag[year >= syear]
    year = year[year >= syear]
    dtime = np.zeros(len(catalog))
    sday = datetime.datetime(int(year[0]), int(month[0]), int(day[0]), int(hour[0]), int(minute[0]), 0)
    '''
    这一段有点小问题先不用， 需要考虑false变为-1还是0，加上去不好
    hour = (minute == 60).astype(int) + hour
    minute[minute == 60] = 0
    day = (hour == 24) + day
    hour[hour == 24] = 0
    '''

    for i in range(len(catalog)):
        eday = datetime.datetime(
            int(year[i]), int(month[i]), int(day[i]), int(hour[i]), int(minute[0]), 0
        )
        dtime[i] = (eday - sday).days + (eday - sday).seconds / 86400  # 86400秒为一天

    # 把重叠的时间去掉,给相同时间的第二个地震加0.001时间
    while np.sum(dtime[1:] - dtime[:-1] <= 0):
        print('↓')
        print(np.sum(dtime[1:] - dtime[:-1] <= 0))
        print('↑')
        for i in range(len(catalog) - 1):
            if dtime[i + 1] <= dtime[i]:
                dtime[i + 1] = dtime[i + 1] + 0.001

    # 把相同的地点取消掉
    index = np.unique(long, return_counts=True)
    is_rep = np.isin(long, index[0][index[1] != 1])
    rounding_error = np.random.uniform(-5e-3, 5e-3, (len(long)))
    rounding_error = rounding_error * is_rep
    long = long + rounding_error

    index = np.unique(lat, return_counts=True)
    is_rep = np.isin(lat, index[0][index[1] != 1])
    rounding_error = np.random.uniform(-5e-3, 5e-3, (len(long)))
    rounding_error = rounding_error * is_rep
    lat = lat + rounding_error
    number = np.linspace(1, len(long), len(long)).astype(int)
    depth = np.zeros(len(long))
    catalog_etas_linux = np.vstack((long, lat, mag, dtime, depth, year, month, day)).T
    etas_in = pd.DataFrame(catalog_etas_linux).reset_index()
    etas_in['index'] = etas_in['index'] + 1
    etas_in.to_csv(aim_filename, sep=' ', header=None, index=None)
    return

region = 'southern_california'
Rcatalog_filename = "data/" + region + "/etasR.csv"
linux_etas_catalog(Rcatalog_filename, region=region, aim_filename='data/' + region + '/' + region + '4.etas')

region = 'INGV'
catalog_filename = 'data/' + region + "/INGV_data_for_etasR.txt"
pd_catalog = pd.read_csv(catalog_filename, sep=',')
linux_etas_catalog(catalog_filename, region=region, aim_filename='data/' + region + '/' + region + '.etas', thm=threshold_mag, sep=',')

def create_etas_in(region, catalog_name='3.etas'):
    data_path = 'data/' + region + '/'
    catalog = pd.read_csv(data_path + region + catalog_name, sep=' ', header=None)
    num = len(catalog)
    thm = np.min(catalog[3])
    f = open(data_path + region + '3.in', 'w')
    f.write('\'' + region + catalog_name + '\'\n')
    f.write('*\n')

    update = 1  # 需要更新参数
    time_start = 0

    time_intervals = np.ceil(np.max(catalog[4]))
    f.write(str(time_intervals))
    f.write('\n')
    f.write(str(num) + '\n' + '0.0\n')
    f.write(str(thm) + '    ' + str(time_start) + '\n')
    '''
    # following is for SC
    f.write('5\n')
    f.write('-113.95 31.01\n-114.05 37.01\n-122.05 37\n-121 35\n-118 31.9\n')
    f.write('751 501 -121.05 -114 31.09 37.01 \n')
    f.write('0.5 0.58 0.617E-02 0.113E+01 0.11E+01 0.156E-04 0.163E+01 0.7533\n')
    '''
    # following is for Italy
    f.write('6\n')
    f.write('12 35\n20 35\n20 44\n16 48\n6 48\n6 42\n')
    f.write('701 601  5 20 35 48\n')
    f.write('0.5 0.2 0.05 2.7 1.2 0.02 2.3 0.3\n')


    #####
    f.write('3\n0.02\n')

    f.write(str(update))

    f.close()
    return

create_etas_in(region, catalog_name='.etas')
