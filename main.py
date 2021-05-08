# %%import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as con
import cv2 as cv
from models import SpinHallDataSet as sh, Lambda4OrientationDataSet as lo, GoldDielectricFunction as gd, \
    SpinHallDataSetFp as fp

# %%declare constants
# center = [(1238, 1011), (1201, 1128), (1203, 1130), (1199, 1123), (1234, 995), (1302, 1046)]
# k_0_NA = 12.07  # in $\mu m^{-1}$
# r_NA = 512  # 502   512
# dataset_number = 4
# max_radius = 600
# spin_hall_angle = np.deg2rad(0)
# angle_width = np.deg2rad(135)  # np.deg2rad(30)
# angle_gap = np.deg2rad(45)  # np.deg2rad(5)
# min_rad = 429
# max_rad = 450  # 502#450


# %%load data
# sh_data_set = sh.SpinHallDataSet('data/spin_hall/{0}'.format(dataset_number), center[dataset_number], max_radius,
#                                k_0_NA, r_NA, 45.3, lamb_offset=2)
#fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)
# lo_data_set = lo.Lambda4OrientationDataSet('data/polorientation.csv')
# gold_dielectric_function = gd.GoldDielectricFunction('data/Olmon_PRB2012_EV.dat')

def fig_2_2(scale):
    gold_dielectric_function = gd.GoldDielectricFunction('data/Olmon_PRB2012_EV.dat')
    fig, ax = plt.subplots(figsize=(6.4 / scale, 4.8 / scale))
    gold_dielectric_function.full_plot(fig, ax)
    plt.show()
    fig.savefig('results/2_2_dispersion.pdf')


def fig_3_2(scale):
    lo_data_set = lo.Lambda4OrientationDataSet('data/polorientation.csv')
    jones_vector, dia_offset = lo_data_set.fit()
    # jones_vector = JonesVector([1, -2])
    # dia_offset = 0
    fig4, axs4 = plt.subplots(2, 1, constrained_layout=True, figsize=(6.4 / scale, 4.8 / scale))
    # fig4.suptitle('diattenuator angle = {:.1f}'.format(np.rad2deg(dia_offset)))
    jones_vector.plot_ellipsis(fig4, axs4[1])
    print(jones_vector)
    print(f'ellipticity{jones_vector.ellipticity}')
    print(f'alpha{jones_vector.alpha}')
    print(f'intensity{jones_vector.intensity}')
    print(f'delta{jones_vector.delta}')
    lo_data_set.plot_jones_fit(jones_vector, dia_offset, fig4, axs4[0])

    fig4.savefig('results/3_2_graph_polarimeter.pdf')
    fig4.show()



def fig_4_1(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600

    # %%load data
    sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)

    fig, ax = plt.subplots(figsize=(6.4 / scale, 4.8 / scale))
    sh_data_set.plot_radial_profile(0, fig, ax)
    fig.show()
    fig.savefig('results/4_1_radial_int.pdf')

    fig1, ax1 = plt.subplots(figsize=(6.4 / scale, 4.8 / scale))
    # fig.set_figsize
    sh_data_set.plot_lorenz_fit_radial_profile(0, (10, 11), fig1, ax1)
    fig1.tight_layout()
    fig1.savefig('results/4_1_lorenz_profile.pdf')
    fig1.show()

    fig2 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax2 = fig2.add_subplot(111, projection='polar')
    #sh_data_set.plot_fp_diff(45, 135, fig, ax)
    fp_data_set.plot_polar(0, fig2, ax2)
    fig2.show()
    fig2.savefig('results/4_1_fp_0.png', dpi=300)

    fig3 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax3 = fig3.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(0, fig3, ax3)
    fig3.legend()
    fig3.show()
    fig3.savefig('results/4_1_bfp_0.png', dpi=300)


def fig_4_3(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600

    # %%load data
    sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)


    fig1 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fp_data_set.plot_polar(45, fig1, ax1)
    fig1.show()
    fig1.savefig('results/4_3_fp_45.png', dpi=300)

    fig2 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax2 = fig2.add_subplot(111, projection='polar')
    fp_data_set.plot_polar(135, fig2, ax2)
    fig2.show()
    fig2.savefig('results/4_3_fp_135.png', dpi=300)

    fig3 = plt.figure(3,figsize=(6.4 / scale, 4.8 / scale))
    ax3 = fig3.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(45, fig3, ax3, zoom=True)
    fig3.show()
    fig3.savefig('results/4_3_bfp_45.png', dpi=300)

    fig4 = plt.figure(4,figsize=(6.4 / scale, 4.8 / scale))
    ax4 = fig4.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(135, fig4, ax4, zoom=True)
    fig4.show()
    fig4.savefig('results/4_3_bfp_135.png', dpi=300)

def fig_4_4(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600

    # %%load data
    #sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
    #                                 k_0_NA, r_NA, 45.3, lamb_offset=2)
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)

    def plot(fig1, ax1, fig2, ax2, angle_width, angle_gap, filename):

        fp_data_set.plot_polar_diff(45, 135, fig1, ax1)
        fp_data_set.plot_masks(0, np.deg2rad(angle_width), np.deg2rad(angle_gap), 0, 150, fig1, ax1)
        fig1.show()
        fig1.savefig(f'results/fig_4_4_fp_{filename}.png', dpi=300)

        fp_data_set.plot_integrated_intensity(0, np.deg2rad(angle_width), np.deg2rad(angle_gap), 0, 150, fig2, ax2)
        fp_data_set.plot_polarisation_marks(fig2, ax2)
        ax2.legend(loc='upper right')
        fig2.tight_layout()
        fig2.show()
        fig2.savefig(f'results/fig_4_4_fp_{filename}_int.pdf')

    fig1 = plt.figure(1, figsize=(6.4/scale, 4.8/scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fig2 = plt.figure(2, figsize=(6.4/scale, 4.8/scale))
    ax2 = fig2.add_subplot(111)
    plot(fig1, ax1, fig2, ax2, 170, 135, 'back')

    fig3 = plt.figure(3, figsize=(6.4 / scale, 4.8 / scale))
    ax3 = fig3.add_subplot(111, projection='polar')
    fig4 = plt.figure(4, figsize=(6.4 / scale, 4.8 / scale))
    ax4 = fig4.add_subplot(111)
    plot(fig3, ax3, fig4, ax4, 135, 45, 'mid')

    fig5 = plt.figure(5, figsize=(6.4 / scale, 4.8 / scale))
    ax5 = fig5.add_subplot(111, projection='polar')
    fig6 = plt.figure(6, figsize=(6.4 / scale, 4.8 / scale))
    ax6 = fig6.add_subplot(111)
    plot(fig5, ax5, fig6, ax6, 45, 10, 'front')

def fig_4_5(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600


    sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    def plot(fig1, ax1, fig2, ax2, angle_width, angle_gap, filename):
        min_rad = 429
        max_rad = 460  # 502#450
        sh_data_set.plot_polar_diff(45, 135, fig1, ax1)
        sh_data_set.plot_masks(0, np.deg2rad(angle_width), np.deg2rad(angle_gap), min_rad, max_rad, fig1, ax1)
        fig1.show()
        fig1.savefig(f'results/fig_4_5_bfp_{filename}.png', dpi=300)

        sh_data_set.plot_integrated_intensity(0, np.deg2rad(angle_width), np.deg2rad(angle_gap), min_rad, max_rad, fig2, ax2)
        sh_data_set.plot_polarisation_marks(fig2, ax2)
        ax2.legend(loc='upper right')
        fig2.tight_layout()
        fig2.show()
        fig2.savefig(f'results/fig_4_5_bfp_{filename}_int.pdf')

    fig1 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fig2 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax2 = fig2.add_subplot(111)
    plot(fig1, ax1, fig2, ax2, 170, 135, 'back')

    fig3 = plt.figure(3, figsize=(6.4 / scale, 4.8 / scale))
    ax3 = fig3.add_subplot(111, projection='polar')
    fig4 = plt.figure(4, figsize=(6.4 / scale, 4.8 / scale))
    ax4 = fig4.add_subplot(111)
    plot(fig3, ax3, fig4, ax4, 135, 45, 'mid')

    fig5 = plt.figure(5, figsize=(6.4 / scale, 4.8 / scale))
    ax5 = fig5.add_subplot(111, projection='polar')
    fig6 = plt.figure(6, figsize=(6.4 / scale, 4.8 / scale))
    ax6 = fig6.add_subplot(111)
    plot(fig5, ax5, fig6, ax6, 45, 10, 'front')

def fig_4_7(scale):
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)
    fig = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax = fig.add_subplot(111, projection='polar')
    fp_data_set.plot_angular_profile(45, 0, 150, fig, ax)
    fp_data_set.plot_angular_profile(135, 0, 150, fig, ax)
    fig.legend()
    fig.show()
    fig.savefig('results/4_7_fp_angular_distribution_45_135.pdf')

    fig1 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fp_data_set.plot_angular_profile_diff(45, 135, 0, 150, fig1, ax1)
    fig1.legend()
    fig1.show()
    fig1.savefig('results/4_7_fp_angular_distribution_diff_45_135.pdf')

def fig_4_8(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600
    min_rad = 429
    max_rad = 460
    sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    fig = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax = fig.add_subplot(111, projection='polar')
    sh_data_set.plot_angular_profile(45, min_rad, max_rad, fig, ax)
    sh_data_set.plot_angular_profile(135, min_rad, max_rad, fig, ax)
    fig.legend()
    fig.show()
    fig.savefig('results/4_8_bfp_angular_distribution_45_135.pdf')

    fig1 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    sh_data_set.plot_angular_profile_diff(45, 135, min_rad, max_rad, fig1, ax1)
    fig1.legend()
    fig1.show()
    fig1.savefig('results/4_8_bfp_angular_distribution_diff_45_135.pdf')

def fig_B_1(scale):
    center = (1302, 1046)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 500
    max_radius = 600

    sh_data_set = sh.SpinHallDataSet('data/spin_hall/5', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/5/fp', (1268, 1204), 250, 45.3, lamb_offset=0)

    fig1 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fp_data_set.plot_polar(0, fig1, ax1)
    fig1.show()
    fig1.savefig('results/B_1_fp_0.png', dpi=300)

    fig2 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax2 = fig2.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(0, fig2, ax2, zoom=False)
    fig2.show()
    fig2.savefig('results/B_1_bfp_90.png', dpi=300)

    fig3, ax3 = plt.subplots(figsize=(6.4 / scale, 4.8 / scale))
    sh_data_set.plot_radial_profile(0, fig3, ax3)
    fig3.show()
    fig3.savefig('results/B_1_radial_int.pdf')

    fig4, ax4 = plt.subplots(figsize=(6.4 / scale, 4.8 / scale))
    # fig.set_figsize
    sh_data_set.plot_lorenz_fit_radial_profile(0, (10.2, 11.7), fig4, ax4)
    fig4.tight_layout()
    fig4.savefig('results/B_1_lorenz_profile.pdf')
    fig4.show()

def fig_B_2(scale):
    center = (1234, 995)
    k_0_NA = 12.07  # in $\mu m^{-1}$
    r_NA = 512
    max_radius = 600

    # %%load data
    sh_data_set = sh.SpinHallDataSet('data/spin_hall/4', center, max_radius,
                                     k_0_NA, r_NA, 45.3, lamb_offset=2)
    fp_data_set = fp.SpinHallDataSetFp('data/spin_hall/4/fp', (1426, 1066), 250, 45.3, lamb_offset=0)


    fig1 = plt.figure(1, figsize=(6.4 / scale, 4.8 / scale))
    ax1 = fig1.add_subplot(111, projection='polar')
    fp_data_set.plot_polar(0, fig1, ax1)
    fig1.show()
    fig1.savefig('results/B_2_fp_0.png', dpi=300)

    fig2 = plt.figure(2, figsize=(6.4 / scale, 4.8 / scale))
    ax2 = fig2.add_subplot(111, projection='polar')
    fp_data_set.plot_polar(90, fig2, ax2)
    fig2.show()
    fig2.savefig('results/B_2_fp_90.png', dpi=300)

    fig3 = plt.figure(3, figsize=(6.4 / scale, 4.8 / scale))
    ax3 = fig3.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(0, fig3, ax3, zoom=True)
    fig3.show()
    fig3.savefig('results/B_2_bfp_0.png', dpi=300)

    fig4 = plt.figure(4, figsize=(6.4 / scale, 4.8 / scale))
    ax4 = fig4.add_subplot(111, projection='polar')
    sh_data_set.plot_polar(90, fig4, ax4, zoom=True)
    fig4.show()
    fig4.savefig('results/B_2_bfp_90.png', dpi=300)
fig_2_2(1.3)
#fig_3_2(1.3)
#fig_4_1(1.3)
#fig_4_3(1.3)
#fig_4_4(1.3)
#fig_4_5(1.3)
#fig_4_7(1.3)
#fig_4_8(1.3)
#fig_B_1(1.3)
#fig_B_2(1.3)