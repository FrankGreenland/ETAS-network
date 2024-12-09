import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import jn, struve, jn_zeros
from cmcrameri import cm
import math

ticks_size = 13
label_size = 15

# Create a canvas
figure = plt.figure(figsize=(15, 12), dpi=300)
# Create a layer
figure_ax1 = figure.add_axes([0.07, 0.55, 0.40, 0.40])  # Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([0.55, 0.55, 0.40, 0.40])  # Sets the position of the layer on the canvas
figure_ax3 = figure.add_axes([0.07, 0.08, 0.40, 0.40])
figure_ax4 = figure.add_axes([0.55, 0.08, 0.40, 0.40])
figure_ax1.annotate('(a)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax2.annotate('(b)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax3.annotate('(c)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax4.annotate('(d)', xy=(-0.12, 1.05), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")

fpath = "plot_community_resolution"
xx = ['0.01', '0.05', '0.1', '0.5', '1']
position = [0.1, 0.3, 0.5, 0.7, 0.9]
bar_color = 'royalblue'
notation_color = 'goldenrod'
### figure a ###
region = 'shanxi'
y1 = np.array(pd.read_csv(f"{fpath}/{region}_percent.dat", header=None)).reshape(-1)
num = np.array(pd.read_csv(f"{fpath}/{region}_community_num.dat", header=None)).reshape(-1)
img11 = figure_ax1.bar(xx, y1, color='royalblue', width=0.5)
figure_ax1.bar_label(img11, color='black', fontsize=label_size-2)
img22 = figure_ax1.bar(xx, num / 200, color='brown', width=0.5)
figure_ax1.bar_label(img22, color=notation_color, fontsize=label_size, labels=num)
figure_ax1.legend([img11, img22], ['Same-community Offspring Proportion', 'Number of communities'], loc='upper right')
figure_ax1.set_ylim([0.0, 1.18])
'''
for ii in range(0, len(position)):
    figure_ax1.annotate(text=str(num[ii]), xycoords='axes fraction', xytext=(position[ii], 0.5),
                        fontsize=label_size, color=notation_color, xy=(0, 0), ha='center')
'''
# figure_ax1.text(6, 1.01, s='test')
# plot
# for i, name in enumerate(model_name):
#   figure_ax1.annotate(name, (d_2D[i],d3D_d2D__3[i]), textcoords='offset points', xytext=(10,-10), ha='left', fontsize=ticks_size)
# figure_ax1.set_ylim([0.5, 1.1])

### figure b ###
region = 'INGV'
y1 = np.array(pd.read_csv(f"{fpath}/{region}_percent.dat", header=None)).reshape(-1)
num = np.array(pd.read_csv(f"{fpath}/{region}_community_num.dat", header=None)).reshape(-1)
img11 = figure_ax2.bar(xx, y1, color='royalblue', width=0.5)
figure_ax2.bar_label(img11, color='black', fontsize=label_size-2)
img22 = figure_ax2.bar(xx, num / 200, color='brown', width=0.5)
figure_ax2.bar_label(img22, color=notation_color, fontsize=label_size, labels=num)
figure_ax2.legend([img11, img22], ['Same-community Offspring Proportion', 'Number of communities'], loc='upper right')
figure_ax2.set_ylim([0.0, 1.18])
'''
for ii in range(0, len(position)):
    figure_ax2.annotate(text=str(num[ii]), xycoords='axes fraction', xytext=(position[ii], 0.5),
                        fontsize=label_size, color=notation_color, xy=(0, 0), ha='center')
'''
### figure c ###
region = 'chuandian'
y1 = np.array(pd.read_csv(f"{fpath}/{region}_percent.dat", header=None)).reshape(-1)
num = np.array(pd.read_csv(f"{fpath}/{region}_community_num.dat", header=None)).reshape(-1)
img11 = figure_ax3.bar(xx, y1, color='royalblue', width=0.5)
figure_ax3.bar_label(img11, color='black', fontsize=label_size-2)
img22 = figure_ax3.bar(xx, num / 200, color='brown', width=0.5)
figure_ax3.bar_label(img22, color=notation_color, fontsize=label_size, labels=num)
figure_ax3.legend([img11, img22], ['Same-community Offspring Proportion', 'Number of communities'], loc='upper right')
figure_ax3.set_ylim([0.0, 1.18])
### figure d ###
region = 'southern_california'
y1 = np.array(pd.read_csv(f"{fpath}/{region}_percent.dat", header=None)).reshape(-1)
num = np.array(pd.read_csv(f"{fpath}/{region}_community_num.dat", header=None)).reshape(-1)
img11 = figure_ax4.bar(xx, y1, color='royalblue', width=0.5)
figure_ax4.bar_label(img11, color='black', fontsize=label_size-2)
img22 = figure_ax4.bar(xx, num / 200, color='brown', width=0.5)
figure_ax4.bar_label(img22, color=notation_color, fontsize=label_size, labels=num)
figure_ax4.legend([img11, img22], ['Same-community Offspring Proportion', 'Number of communities'], loc='upper right')
figure_ax4.set_ylim([0.0, 1.18])


figure_ax1.set_xlabel("resolution parameter " + r'$\gamma$', fontsize=label_size)
figure_ax1.set_ylabel("Same-community Offspring Proportion", fontsize=label_size)
figure_ax1.tick_params(labelsize=ticks_size)
figure_ax1.grid(which='both', linestyle=':', axis='y')
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax1.set_axisbelow(True)
figure_ax1.minorticks_on()

figure_ax2.set_xlabel(r"resolution parameter " + r'$\gamma$', fontsize=label_size)
figure_ax2.set_ylabel("Same-community Offspring Proportion", fontsize=label_size)
figure_ax2.tick_params(labelsize=ticks_size)
figure_ax2.grid(which='both', linestyle=':', axis='y')
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax2.set_axisbelow(True)
figure_ax2.minorticks_on()

figure_ax3.set_xlabel("resolution parameter " + r'$\gamma$', fontsize=label_size)
figure_ax3.set_ylabel("Same-community Offspring Proportion", fontsize=label_size)
figure_ax3.tick_params(labelsize=ticks_size)
# figure_ax3.grid(which='major', linestyle='--')
figure_ax3.grid(which='both', linestyle=':', axis='y')
# figure_ax3.tick_params(axis='x', labelrotation=-22.8, grid_linestyle='-')
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_on()

figure_ax4.set_xlabel("resolution parameter " + r'$\gamma$', fontsize=label_size)
figure_ax4.set_ylabel("Same-community Offspring Proportion", fontsize=label_size)
figure_ax4.tick_params(labelsize=ticks_size)
figure_ax4.grid(which='both', linestyle=':', axis='y')
# figure_ax4.grid(which='major', linestyle='--')
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax4.set_axisbelow(True)
figure_ax4.minorticks_on()

# Displays the drawing results
# plt.savefig("./Figure_2.svg", dpi=300, format="svg")
plt.savefig(f"{fpath}/percent.png", dpi=300, format="png")
# plt.ioff() # Open the interactive drawing. If closed, the program pauses while drawing
