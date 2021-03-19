# %%import libraries
import matplotlib.pyplot as plt
import numpy as np
from models import SpinHallDataSet as sh, Lambda4OrientationDataSet as lo

# %%declare constants
center = [(1238, 1011), (1201, 1128), (1203, 1130), (1199, 1123)]
k_0_NA = 12.07  # in \mu m^{-1}
r_NA = 504
dataset_number = 0
max_radius = 600
spin_hall_angle = 0# 0.15
angle_width = 3 / 4 * np.pi
angle_gap = np.pi / 4
min_rad = 425
max_rad = 448

# %%load data
#sh_data_set = sh.SpinHallDataSet('data/spin_hall/{0}'.format(dataset_number), center[dataset_number], max_radius,
#                                 k_0_NA, r_NA)
lo_data_set = lo.Lambda4OrientationDataSet('data/polorientation.csv')


# %%plot
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111, projection='polar')
# sh_data_set.plot_polar_diff(45, 135, fig1, ax1)
#
# sh_data_set.plot_masks(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig1, ax1)
# legend_angle = np.deg2rad(67.5)
# fig1.legend(loc="lower left")
# fig1.savefig('results/polar_diff_mask{0}.png'.format(dataset_number), dpi=300)
# fig1.show()
#
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# sh_data_set.plot_integrated_intensity(spin_hall_angle, angle_width, angle_gap, min_rad, max_rad, fig2, ax2)
# ax3 = ax2.twinx()
# lo_data_set.plot_orientation(fig2, ax3)
# fig2.tight_layout
# fig2.legend(loc="upper left", mode="expand", ncol=2)
# fig2.savefig('results/integrated_intesity{0}.png'.format(dataset_number), dpi=300)
# fig2.show()


result = lo_data_set.fit()
Ex, Ex_j, Ey, Ey_j = result.x
resultJonesVector = np.array([Ex + 1j * Ex_j, Ey + 1j * Ey_j])
fig, ax = plt.subplots()
#lo_data_set.plot_jones_vector_ellipse(resultJonesVector, fig, ax)
#vectors = lo_data_set.polarimeter_simulation(resultJonesVector)
lo_data_set.plot_jones_fit(resultJonesVector, fig, ax)
#lo_data_set.plot_jones_vector_ellipse(resultJonesVector, fig, ax)
plt.show()
