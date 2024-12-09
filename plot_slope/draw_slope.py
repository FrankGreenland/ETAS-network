import matplotlib.pyplot as plt
import numpy as np
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

### figure a ###
# !!
x1 = np.load('./plot_slope/shanxi_x.npy')
y1 = np.load('./plot_slope/shanxi_y.npy')
logeE = np.log(1 / x1)
logeN = np.log(y1)
nn = 5
slope, intercept = np.polyfit(logeE[:nn], logeN[:nn], 1)
img11 = figure_ax1.scatter(logeE, logeN, color='black', s=200, linewidths=2, marker='+')
img12, = figure_ax1.plot(logeE, slope * logeE + intercept, color='black', linestyle='dashdot')
# plot
# for i, name in enumerate(model_name):
#   figure_ax1.annotate(name, (d_2D[i],d3D_d2D__3[i]), textcoords='offset points', xytext=(10,-10), ha='left', fontsize=ticks_size)

figure_ax1.text(-1.5, 4.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax1.legend([img11, img12], ['Different Cell Length', r'Fit Line: y = ax + b (for $\epsilon<10)$'], handlelength=4,
                  loc='upper left', fontsize=ticks_size)
figure_ax1.set_xlim([-5, 0.5])
figure_ax1.set_ylim([2, 10.0])

### figure b ###
x2 = np.load('./plot_slope/INGV_x.npy')
y2 = np.load('./plot_slope/INGV_y.npy')
logeE = np.log(1 / x2)
logeN = np.log(y2)
nn = 5
slope, intercept = np.polyfit(logeE[:nn], logeN[:nn], 1)

img21 = figure_ax2.scatter(logeE, logeN, color='black', s=200, linewidths=2, marker='+')
img22, = figure_ax2.plot(logeE, slope * logeE + intercept, color='black', linestyle='dashdot')
figure_ax2.text(-1.5, 5.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax2.legend([img21, img22], ['Different Cell Length', r'Fit Line: y = ax + b (for $\epsilon<10)$'],
                  handlelength=4, loc='upper left', fontsize=ticks_size)
figure_ax2.set_xlim([-5, 0.5])
figure_ax2.set_ylim([3, 11.0])


### figure c ###
x3 = np.load('./plot_slope/chuandian_x.npy')
y3 = np.load('./plot_slope/chuandian_y.npy')
logeE = np.log(1 / x3)
logeN = np.log(y3)
nn = 5
slope, intercept = np.polyfit(logeE[:nn], logeN[:nn], 1)

img31 = figure_ax3.scatter(logeE, logeN, color='black', s=200, linewidths=2, marker='+')
img32, = figure_ax3.plot(logeE, slope * logeE + intercept, color='black', linestyle='dashdot')
figure_ax3.text(-1.5, 5.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax3.legend([img31, img32], ['Different Cell Length', r'Fit Line: y = ax + b (for $\epsilon<10)$'],
                  handlelength=4, loc='upper left', fontsize=ticks_size)
figure_ax3.set_xlim([-5, 0.5])
figure_ax3.set_ylim([3, 11.0])

### figure d ###
x4 = np.load('./plot_slope/southern_california_x.npy')
y4 = np.load('./plot_slope/southern_california_y.npy')
logeE = np.log(1 / x4)
logeN = np.log(y4)
nn = 5
slope, intercept = np.polyfit(logeE[:nn], logeN[:nn], 1)
img41 = figure_ax4.scatter(logeE, logeN, color='black', s=200, linewidths=2, marker='+')
img42,  = figure_ax4.plot(logeE, slope * logeE + intercept, color='black', linestyle='dashdot')
figure_ax4.text(-1.5, 5.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax4.legend([img31, img32], ['Different Cell Length', r'Fit Line: y = ax + b (for $\epsilon<10)$'], handlelength=4,loc='upper left', fontsize=ticks_size)
figure_ax4.set_xlim([-5, 0.5])
figure_ax4.set_ylim([2, 10.0])


figure_ax1.set_xlabel(r'$-ln\epsilon$', fontsize=label_size)
figure_ax1.set_ylabel(r'$lnN$', fontsize=label_size)
figure_ax1.tick_params(labelsize=ticks_size)
figure_ax1.grid(which='both', linestyle=':')
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax1.set_axisbelow(True)
figure_ax1.minorticks_on()

figure_ax2.set_xlabel(r'$-ln\epsilon$', fontsize=label_size)
figure_ax2.set_ylabel(r'$lnN$', fontsize=label_size)
figure_ax2.tick_params(labelsize=ticks_size)
figure_ax2.grid(which='both', linestyle=':')
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax2.set_axisbelow(True)
figure_ax2.minorticks_on()

figure_ax3.set_xlabel(r'$-ln\epsilon$', fontsize=label_size)
figure_ax3.set_ylabel(r'$lnN$', fontsize=label_size)
figure_ax3.tick_params(labelsize=ticks_size)
# figure_ax3.grid(which='major', linestyle='--')
figure_ax3.grid(which='both', linestyle=':')
# figure_ax3.tick_params(axis='x', labelrotation=-22.8, grid_linestyle='-')
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_on()

figure_ax4.set_xlabel(r'$-ln\epsilon$', fontsize=label_size)
figure_ax4.set_ylabel(r'$lnN$', fontsize=label_size)
figure_ax4.tick_params(labelsize=ticks_size)
figure_ax4.grid(which='both', linestyle=':')
# figure_ax4.grid(which='major', linestyle='--')
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax4.set_axisbelow(True)
figure_ax4.minorticks_on()

# Displays the drawing results
# plt.savefig("./Figure_2.svg", dpi=300, format="svg")
plt.savefig("./fitting_line.png", dpi=300, format="png")
# plt.ioff() # Open the interactive drawing. If closed, the program pauses while drawing
