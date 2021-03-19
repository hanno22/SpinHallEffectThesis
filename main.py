# %%import libraries
import matplotlib.pyplot as plt
import numpy as np

from models import SpinHallDataSet as sh, Lambda4OrientationDataSet as lo

# %%declare constants
center = [(1238, 1011), (1201, 1128), (1203, 1130), (1199, 1123)]
k_0_NA = 12.07  # in $\mu m^{-1}$
r_NA = 504
dataset_number = 0
max_radius = 600
spin_hall_angle = 0  # 0.15
angle_width = 3 / 4 * np.pi
angle_gap = np.pi / 4
min_rad = 425
max_rad = 448

# %%load data
sh_data_set = sh.SpinHallDataSet('data/spin_hall/{0}'.format(dataset_number), center[dataset_number], max_radius,
                                 k_0_NA, r_NA)
lo_data_set = lo.Lambda4OrientationDataSet('data/polorientation.csv')

# %%plot spin_hall_data
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='polar')
sh_data_set.plot_polar_diff(45, 135, fig1, ax1)

sh_data_set.plot_masks(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig1, ax1)
legend_angle = np.deg2rad(67.5)
fig1.legend(loc="lower left")
fig1.savefig('results/polar_diff_mask{0}.png'.format(dataset_number), dpi=300)
fig1.show()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
sh_data_set.plot_integrated_intensity(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig2, ax2)
ax3 = ax2.twinx()
lo_data_set.plot_orientation(fig2, ax3)
fig2.legend(loc="upper left", mode="expand", ncol=2)
fig2.savefig('results/integrated_intesity{0}.png'.format(dataset_number), dpi=300)
fig2.show()

# %%plot Polarimeterdata
jones_vector, dia_offset = lo_data_set.fit()
# jones_vector = JonesVector([1, -2])
# dia_offset = 0
fig4, axs4 = plt.subplots(2, 1, constrained_layout=True)
fig4.suptitle('diattenuator angle = {:.1f}'.format(np.rad2deg(dia_offset)))
jones_vector.plot_ellipsis(fig4, axs4[0])
b = jones_vector.alpha
lo_data_set.plot_jones_fit(jones_vector, dia_offset, fig4, axs4[1])
fig4.savefig('results/polarimeter.png', dpi=300)
fig4.show()
