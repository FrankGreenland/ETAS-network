import pandas as pd
import numpy as np

thm = 2.2
region = 'southern_california'
filepath = '../data/' + region
filename = 'etasR.csv'
catalog = pd.read_csv(filepath + '/' + filename, sep=',', index_col=None)
catalog = catalog[catalog['mag'] >= thm]
position = np.array(catalog)[:, 2:4]

position = pd.DataFrame(position)
position.to_csv(region + '_position', header=None, index=None)



