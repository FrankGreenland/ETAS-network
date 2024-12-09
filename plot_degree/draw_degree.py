import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import jn, struve, jn_zeros
from cmcrameri import cm
import math
from scipy.optimize import curve_fit


def funclinear(x, c, a):
  return -1 * a * x + c


def fittinglinear(xx, yy):
    k = np.log10(xx)
    kprob = np.log10(yy)
    k2 = k[kprob != -1*np.inf]
    kprob2 = kprob[kprob != -1*np.inf]
    # plt.scatter(k2, kprob2, color='b')
    popt, pcov = curve_fit(funclinear, k2, kprob2)
    k3 = np.hstack((np.array([0]), k2))
    y2 = [funclinear(i, popt[0], popt[1]) for i in k3]
    # plt.plot(k2, y2, 'r--')
    # plt.title('c:' + str(popt[0]) + '   a=' + str(popt[1]))
    # plt.show()
    return np.power(10, k3), np.power(10, np.array(y2)), popt[1]


def fittinglinear_norm(xx, yy):
    k = np.log10(xx)
    kprob = np.log10(yy)
    k2 = k[kprob != -1*np.inf]
    kprob2 = kprob[kprob != -1*np.inf]
    # plt.scatter(k2, kprob2, color='b')
    popt, pcov = curve_fit(funclinear, k2, kprob2)
    y2 = [funclinear(i, popt[0], popt[1]) for i in k2]
    # plt.plot(k2, y2, 'r--')
    # plt.title('c:' + str(popt[0]) + '   a=' + str(popt[1]))
    # plt.show()
    return np.power(10, k2), np.power(10, np.array(y2)), popt[1]


ticks_size = 13
label_size = 15

# Create a canvas
figure = plt.figure(figsize=(21, 24), dpi=300)
# Create a layer
x_l = 0.27
y_l = 0.21
figure_ax1 = figure.add_axes([0.06,0.77,x_l, y_l]) #Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([0.39,0.77,x_l, y_l]) #Sets the position of the layer on the canvas
figure_ax3 = figure.add_axes([0.72,0.77,x_l, y_l])
figure_ax4 = figure.add_axes([0.06,0.52,x_l, y_l])
figure_ax5 = figure.add_axes([0.39,0.52,x_l, y_l])
figure_ax6 = figure.add_axes([0.72,0.52,x_l, y_l])
figure_ax7 = figure.add_axes([0.06,0.27,x_l, y_l])
figure_ax8 = figure.add_axes([0.39,0.27,x_l, y_l])
figure_ax9 = figure.add_axes([0.72,0.27,x_l, y_l])
figure_ax10 = figure.add_axes([0.06,0.02,x_l, y_l])
figure_ax11 = figure.add_axes([0.39,0.02,x_l, y_l])
figure_ax12 = figure.add_axes([0.72,0.02,x_l, y_l])

x_d = -0.09
y_d = 1.05
figure_ax1.annotate('(a)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax2.annotate('(b)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax3.annotate('(c)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax4.annotate('(d)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax5.annotate('(e)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax6.annotate('(f)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax7.annotate('(g)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax8.annotate('(h)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax9.annotate('(i)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax10.annotate('(j)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax11.annotate('(k)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax12.annotate('(l)', xy=(x_d, y_d), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")


### figure a ###

x1 = np.load('./plot_degree/shanxi_degree_x.npy')
y1 = np.load('./plot_degree/shanxi_degree_y.npy')
nn = 2 # 不做处理时为0
xline, yline, slope = fittinglinear(x1[nn:], y1[nn:])

img11, = figure_ax1.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax1.scatter(x1, y1, color='black', s=40)
# figure_ax1.text(-1.5, 4.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax1.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
# figure_ax1.set_xlim([10, np.max(x1) + 20])
figure_ax1.set_xlim([np.min(x1[nn:])*0.95, np.max(x1) + 20])
figure_ax1.set_ylim([1e-3, 0.3])

### figure b ###
x1 = np.load('./plot_degree/shanxi_weight_x.npy')
y1 = np.load('./plot_degree/shanxi_weight_y.npy')
nn = 0  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax2.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax2.scatter(x1, y1, color='black', s=40)
figure_ax2.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax2.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax2.set_ylim([1e-5, 1])

### figure c ###
x1 = np.load('./plot_degree/shanxi_strength_x.npy')
y1 = np.load('./plot_degree/shanxi_strength_y.npy')
nn = 0 # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax3.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax3.scatter(x1, y1, color='black', s=40)
figure_ax3.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)

figure_ax3.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax3.set_ylim([5e-5, 1])


### INGV ###
### figure a2 ###
region = 'INGV'
x1 = np.load('./plot_degree/' + region + '_degree_x.npy')
y1 = np.load('./plot_degree/' + region + '_degree_y.npy')
nn = 1  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])

img11, = figure_ax4.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax4.scatter(x1, y1, color='black', s=40)
# figure_ax1.text(-1.5, 4.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax4.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
# figure_ax4.set_xlim([np.min(x1[nn:]), np.max(x1) + 20])
figure_ax4.set_xlim([np.min(x1[nn:])*0.95, np.max(x1) + 20])
figure_ax4.set_ylim([1e-4, 1])

### figure b2 ###
x1 = np.load('./plot_degree/' + region + '_weight_x.npy')
y1 = np.load('./plot_degree/' + region + '_weight_y.npy')
nn = 0  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax5.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax5.scatter(x1, y1, color='black', s=40)
figure_ax5.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax5.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax5.set_ylim([1e-5, 1])

### figure c2 ###
x1 = np.load('./plot_degree/' + region + '_strength_x.npy')
y1 = np.load('./plot_degree/' + region + '_strength_y.npy')
nn = 0 # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax6.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax6.scatter(x1, y1, color='black', s=40)
figure_ax6.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax6.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax6.set_ylim([5e-5, 1])



### chuandian ###
### figure a3 ###
region = 'chuandian'
x1 = np.load('./plot_degree/' + region + '_degree_x.npy')
y1 = np.load('./plot_degree/' + region + '_degree_y.npy')
nn = 1  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])

img11, = figure_ax7.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax7.scatter(x1, y1, color='black', s=40)
# figure_ax1.text(-1.5, 4.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax7.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
# figure_ax4.set_xlim([np.min(x1[nn:]), np.max(x1) + 20])
figure_ax7.set_xlim([np.min(x1[nn:])*0.95, np.max(x1) + 20])
figure_ax7.set_ylim([1e-4, 1])

### figure b3 ###
x1 = np.load('./plot_degree/' + region + '_weight_x.npy')
y1 = np.load('./plot_degree/' + region + '_weight_y.npy')
nn = 0  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax8.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax8.scatter(x1, y1, color='black', s=40)
figure_ax8.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax8.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax8.set_ylim([1e-5, 1])

### figure c3 ###
x1 = np.load('./plot_degree/' + region + '_strength_x.npy')
y1 = np.load('./plot_degree/' + region + '_strength_y.npy')
nn = 0 # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax9.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax9.scatter(x1, y1, color='black', s=40)
figure_ax9.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax9.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax9.set_ylim([5e-5, 1])


### SC ###
### figure a4 ###
region = 'southern_california'
x1 = np.load('./plot_degree/' + region + '_degree_x.npy')
y1 = np.load('./plot_degree/' + region + '_degree_y.npy')
nn = 1  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])

img11, = figure_ax10.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax10.scatter(x1, y1, color='black', s=40)
# figure_ax1.text(-1.5, 4.5, f'Slope: {slope:.3e}\n', style='oblique', fontsize=label_size, color="black")
figure_ax10.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
# figure_ax4.set_xlim([np.min(x1[nn:]), np.max(x1) + 20])
figure_ax10.set_xlim([np.min(x1[nn:])*0.95, np.max(x1) + 20])
figure_ax10.set_ylim([1e-4, 1])

### figure b4 ###
x1 = np.load('./plot_degree/' + region + '_weight_x.npy')
y1 = np.load('./plot_degree/' + region + '_weight_y.npy')
nn = 0  # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax11.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax11.scatter(x1, y1, color='black', s=40)
figure_ax11.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax11.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax11.set_ylim([1e-5, 1])

### figure c4 ###
x1 = np.load('./plot_degree/' + region + '_strength_x.npy')
y1 = np.load('./plot_degree/' + region + '_strength_y.npy')
nn = 0 # 不做处理时为0
xline, yline, slope = fittinglinear_norm(x1[nn:], y1[nn:])
img11, = figure_ax12.loglog(xline, yline, color='red', linestyle='dashdot')
img12 = figure_ax12.scatter(x1, y1, color='black', s=40)
figure_ax12.legend([img11], [f'Fit Line: y = ax + b (slope={slope:.3})'], handlelength=4,
                  loc='upper right', fontsize=ticks_size)
figure_ax12.set_xlim([np.min(x1)*0.9, np.max(x1)*1.1])
figure_ax12.set_ylim([5e-5, 1])


#############
figure_ax1.set_xlabel(r'$k$', fontsize=label_size)
figure_ax1.set_ylabel(r'P$(k)$', fontsize=label_size)
figure_ax1.tick_params(labelsize=ticks_size)
figure_ax1.grid(which='both', linestyle=':')
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax1.set_axisbelow(True)
figure_ax1.minorticks_on()

figure_ax2.set_xlabel(r'$w$', fontsize = label_size)
figure_ax2.set_ylabel(r'P$(w)$', fontsize = label_size)
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.grid(which='both', linestyle=':')
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax2.set_axisbelow(True)
figure_ax2.minorticks_on()


figure_ax3.set_xlabel(r's', fontsize = label_size)
figure_ax3.set_ylabel(r'P$(s)$', fontsize = label_size)
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax3.grid(which='both', linestyle=':')
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_on()


#### INGV ####
figure_ax4.set_xlabel(r'$k$', fontsize=label_size)
figure_ax4.set_ylabel(r'P$(k)$', fontsize=label_size)
figure_ax4.tick_params(labelsize=ticks_size)
figure_ax4.grid(which='both', linestyle=':')
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax4.set_axisbelow(True)
figure_ax4.minorticks_on()

figure_ax5.set_xlabel(r'$w$', fontsize = label_size)
figure_ax5.set_ylabel(r'P$(w)$', fontsize = label_size)
figure_ax5.tick_params(labelsize = ticks_size)
figure_ax5.grid(which='both', linestyle=':')
figure_ax5.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax5.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax5.set_axisbelow(True)
figure_ax5.minorticks_on()


figure_ax6.set_xlabel(r's', fontsize = label_size)
figure_ax6.set_ylabel(r'P$(s)$', fontsize = label_size)
figure_ax6.tick_params(labelsize = ticks_size)
figure_ax6.grid(which='both', linestyle=':')
figure_ax6.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax6.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax6.set_axisbelow(True)
figure_ax6.minorticks_on()


#### chuandian ####
figure_ax7.set_xlabel(r'$k$', fontsize=label_size)
figure_ax7.set_ylabel(r'P$(k)$', fontsize=label_size)
figure_ax7.tick_params(labelsize=ticks_size)
figure_ax7.grid(which='both', linestyle=':')
figure_ax7.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax7.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax7.set_axisbelow(True)
figure_ax7.minorticks_on()

figure_ax8.set_xlabel(r'$w$', fontsize = label_size)
figure_ax8.set_ylabel(r'P$(w)$', fontsize = label_size)
figure_ax8.tick_params(labelsize = ticks_size)
figure_ax8.grid(which='both', linestyle=':')
figure_ax8.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax8.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax8.set_axisbelow(True)
figure_ax8.minorticks_on()


figure_ax9.set_xlabel(r's', fontsize = label_size)
figure_ax9.set_ylabel(r'P$(s)$', fontsize = label_size)
figure_ax9.tick_params(labelsize = ticks_size)
figure_ax9.grid(which='both', linestyle=':')
figure_ax9.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax9.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax9.set_axisbelow(True)
figure_ax9.minorticks_on()


#### SC ####
figure_ax10.set_xlabel(r'$k$', fontsize=label_size)
figure_ax10.set_ylabel(r'P$(k)$', fontsize=label_size)
figure_ax10.tick_params(labelsize=ticks_size)
figure_ax10.grid(which='both', linestyle=':')
figure_ax10.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax10.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax10.set_axisbelow(True)
figure_ax10.minorticks_on()

figure_ax11.set_xlabel(r'$w$', fontsize = label_size)
figure_ax11.set_ylabel(r'P$(w)$', fontsize = label_size)
figure_ax11.tick_params(labelsize = ticks_size)
figure_ax11.grid(which='both', linestyle=':')
figure_ax11.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax11.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax11.set_axisbelow(True)
figure_ax11.minorticks_on()


figure_ax12.set_xlabel(r's', fontsize = label_size)
figure_ax12.set_ylabel(r'P$(s)$', fontsize = label_size)
figure_ax12.tick_params(labelsize = ticks_size)
figure_ax12.grid(which='both', linestyle=':')
figure_ax12.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax12.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax12.set_axisbelow(True)
figure_ax12.minorticks_on()
# Displays the drawing results
#plt.savefig("./Figure_2.svg", dpi=300, format="svg")
plt.savefig("./plot_degree/degree.png", dpi=300, format="png")
#plt.ioff() # Open the interactive drawing. If closed, the program pauses while drawing
