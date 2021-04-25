# %%import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as con
import cv2 as cv
from models import SpinHallDataSet as sh, Lambda4OrientationDataSet as lo, GoldDielectricFunction as gd

# %%declare constants
center = [(1238, 1011), (1201, 1128), (1203, 1130), (1199, 1123), (1234, 995), (1302, 1046)]
k_0_NA = 12.07  # in $\mu m^{-1}$
r_NA = 502#512
dataset_number = 4
max_radius = 600
spin_hall_angle = np.deg2rad(0)
angle_width = np.deg2rad(30) #np.deg2rad(30)
angle_gap = np.deg2rad(5) #np.deg2rad(5)
min_rad = 429
max_rad = 450

# %%load data
sh_data_set = sh.SpinHallDataSet('data/spin_hall/{0}'.format(dataset_number), center[dataset_number], max_radius,
                              k_0_NA, r_NA, 45.3, lamb_offset=2)
lo_data_set = lo.Lambda4OrientationDataSet('data/polorientation.csv')
#gold_dielectric_function = gd.GoldDielectricFunction('data/Olmon_PRB2012_EV.dat')

def example():

    #print(gold_dielectric_function.eps_interpolated(1.95868))
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(45, fig1, ax1)
    fig1.savefig('results/example_polar.png', dpi=300)
    fig2, ax2 = plt.subplots()
    sh_data_set.plot_radial_profile(0, fig2, ax2)
    fig2.savefig('results/example_radial.pdf')

    fig3, ax3 = plt.subplots()
    fp = cv.cvtColor(cv.imread('data/spin_hall/fp_0.bmp'), cv.COLOR_BGR2GRAY)
    max = np.max(fp)
    scale = 45.3
    x = np.linspace(0, fp.shape[0] / scale, fp.shape[0])
    y = np.linspace(0, fp.shape[1] / scale, fp.shape[1])
    clev = np.arange(0, 1, .01)
    cs = ax3.contourf(x, y, fp / max, clev, cmap='plasma')
    #ax3.set_aspect('equal')
    ax3.set(xlabel='$x / \\mathrm{\\mu m}$', ylabel='$y / \\mathrm{\\mu m}$', aspect='equal')
    cbar = fig3.colorbar(cs)
    cbar.set_label('intensity', rotation=90)
    fig3.tight_layout()
    fig3.savefig('results/example_focal_plane.png', dpi=300)

    fig4, ax4 = plt.subplots()
    sh_data_set.plot_angular_profile(135, fig4, ax4)
    fig4.show()


def dispersion():
    fig, ax = plt.subplots()
    #ax.plot(gold_dielectric_function.E, gold_dielectric_function.eps.real, label='real_eps')
    #ax.plot(gold_dielectric_function.E, gold_dielectric_function.eps.imag, label='imag_eps')
    #ax.plot(gold_dielectric_function.E, gold_dielectric_function.n_eff(1).real, label='n_eff_real')
    #ax.plot(gold_dielectric_function.E, gold_dielectric_function.n_eff(1).imag, label='n_eff_imag')
    gold_dielectric_function.full_plot(fig, ax)
    ax.axhline(1, linestyle ='--', label='n_vakuum')
    ax.legend()
    plt.show()
    # fig.savefig('results/dispersion.png', dpi = 300)



# %%plot spin_hall_data
def spin_hall():
    fig1 = plt.figure(1) #figsize=(6.4/1.2, 4.8/1.2))
    ax1 = fig1.add_subplot(111, projection='polar')
    sh_data_set.plot_polar_diff(45, 135, fig1, ax1)

    sh_data_set.plot_masks(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig1, ax1)
    #legend_angle = np.deg2rad(67.5)
    #fig1.legend(loc="lower left")
    fig1.savefig('results/polar_diff_mask{0}.png'.format(dataset_number), dpi=300)
    fig1.show()

    fig2 = plt.figure(2) #figsize=(6.4/, 4.8/1.5))
    ax2 = fig2.add_subplot(111)
    sh_data_set.plot_integrated_intensity(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig2, ax2)
    sh_data_set.plot_polarisation_marks(fig2, ax2)
    #ax3 = ax2.twinx()
    #lo_data_set.plot_orientation(fig2, ax3)
    #fig2.legend(loc="upper left", mode="expand", ncol=2)
    ax2.legend()
    fig2.savefig('results/integrated_intesity{0}.pdf'.format(dataset_number), dpi=300)
    fig2.show()


def polarimeter():
    jones_vector, dia_offset = lo_data_set.fit()
    # jones_vector = JonesVector([1, -2])
    # dia_offset = 0
    fig4, axs4 = plt.subplots(2, 1, constrained_layout=True)
    #fig4.suptitle('diattenuator angle = {:.1f}'.format(np.rad2deg(dia_offset)))
    jones_vector.plot_ellipsis(fig4, axs4[0])
    print(jones_vector)
    print(f'ellipticity{jones_vector.ellipticity}')
    print(f'alpha{jones_vector.alpha}')
    print(f'intensity{jones_vector.intensity}')
    print(f'delta{jones_vector.delta}')
    lo_data_set.plot_jones_fit(jones_vector, dia_offset, fig4, axs4[1])
    fig4.savefig('results/graph_polarimeter.pdf')
    fig4.show()

def dirt():
    fig1, ax1 = plt.subplots()
    sh_data_set.plot_radial_profile(0, fig1, ax1)
    fig1.savefig('results/dirt_radial.pdf')
    fig1.show()

    fig2 = plt.figure(2)  # figsize=(6.4/1.2, 4.8/1.2))
    ax2 = fig2.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(0, fig2, ax2)
    fig2.savefig('results/dirt_polar.png', dpi=300)

    fig3, ax3 = plt.subplots()
    sh_data_set.plot_lorenz_fit_radial_profile(0, (10, 11.8), fig3, ax3)
    fig3.savefig('results/dirt_lorentz.pdf')
    fig3.show()


def lorentz_plot():
    fig, ax = plt.subplots()
    sh_data_set.plot_lorenz_fit_radial_profile(0, (10, 11), fig, ax)
    fig.savefig('results/lorenz_profile.pdf')
    fig.show()

def polar_plots(angles):
    for ang in angles:
        fig = plt.figure()  # figsize=(6.4/1.2, 4.8/1.2))
        ax = fig.add_subplot(111, projection='polar')
        sh_data_set.plot_polar(ang, fig, ax)
        fig.show()
        fig.savefig('results/polar_{0}.png'.format(ang), dpi=300)

#polar_plots([0,45,90,135])
#example()
#spin_hall()
#dirt()

#polarimeter()
fig1, ax1 = plt.subplots()
#sh_data_set.plot_fp_diff(45, 135, fig, ax)
sh_data_set.plot_fp(0, fig1, ax1)
fig1.savefig('results/fp_0.png', dpi=300)

fig2, ax2 = plt.subplots()
sh_data_set.plot_fp(45, fig2, ax2)
fig2.savefig('results/fp_45.png', dpi=300)

fig3, ax3 = plt.subplots()
sh_data_set.plot_fp(135, fig3, ax3)
fig3.savefig('results/fp_135.png', dpi=300)

fig4, ax4 = plt.subplots()
sh_data_set.plot_fp_diff(110, 160, fig4, ax4)
fig4.savefig('results/diff_fp_110_160.png', dpi=300)

fig5, ax5 = plt.subplots()
sh_data_set.plot_fp(90, fig5, ax5)
fig5.savefig('results/fp_90.png', dpi=300)

