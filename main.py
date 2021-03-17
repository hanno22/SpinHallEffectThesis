# %%import libaries
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import os
import re
pol_data = np.genfromtxt('data/polorientation.csv', delimiter=',')

#center = (1238, 1011)
center = (1201, 1128)
#center = (563, 338)
max_radius = 500
# %%load data
path = 'data/spin_hall/2'
files = sorted(glob.glob(os.path.join(path, '*.bmp')), key=lambda x: float(re.search("([0-9]*).bmp", x).group(1)))
image_angle = np.empty(len(files))
image_dimensions = cv.imread(files[0]).shape[0:2]
source = np.empty((len(files), *image_dimensions))
for i, file in enumerate(files):
    source[i] = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY)
    image_angle[i] = float(re.search("([0-9]*).bmp", file).group(1))

# #%% load test data
# center = (213, 206)
# max_radius = 300
# source = cv.imread('data/spin_hall/polar_remap.png')
# data = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

# %% load test data

image = cv.cvtColor(cv.imread('data/TestFringe.bmp'), cv.COLOR_BGR2GRAY);
#%% functions
def get_polar_coordinates(polar_image):
    rads = np.linspace(0, max_radius, polar_image.shape[1])
    angs = np.linspace(0, 2 * np.pi, polar_image.shape[0])
    return angs, rads

def plot_polar(polar_image):
    angs, rads = get_polar_coordinates(polar_image)
    polar_image_swaped = np.swapaxes(polar_image, 0, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='polar')
    cs = ax1.contourf(angs, rads, polar_image_swaped[:, ::-1], cmap='binary')
    ax1.plot(angs, np.full(angs.shape, 430), linewidth=0.3)
    ax1.plot(angs, np.full(angs.shape, 450), linewidth=0.3)
    ax1.plot(np.full(rads.shape, np.pi / 4), rads)
    fig1.colorbar(cs)
    plt.show()


def plot_radial_profile(polar_image):
    radial_profile = np.average(polar_image, axis=0)
    fig, ax = plt.subplots()
    ax.plot(radial_profile, '.')
    plt.savefig('test.png', dpi=500)

def plot_angular_profile(polar_image):
    angs, rads = get_polar_coordinates(polar_image)
    angular_profile = np.average(polar_image, axis=1)
    fig, ax = plt.subplots()
    ax.plot(angs, angular_profile, '.')
    plt.savefig('test.png', dpi=500)

def integrate_intensity_of_half_spaces(polar_image, mid_angle, angle_width, min_rad, max_rad):
    angs, rads = get_polar_coordinates(polar_image)

    if mid_angle + angle_width < 2 * np.pi:
        left_angle_mask = (angs >= mid_angle) & (angs <= mid_angle + angle_width)
    else:
        left_angle_mask = (angs >= mid_angle) | (angs <= mid_angle + angle_width - (2 * np.pi))

    if mid_angle - angle_width > 0:
        right_angle_mask = (angs <= mid_angle) & (angs >= mid_angle - angle_width)
    else:
        right_angle_mask = (angs <= mid_angle) | (angs >= 2 * np.pi + (mid_angle - angle_width))

    radial_mask = (rads > min_rad) & (rads <max_rad)
    masked_image = polar_image[:, radial_mask]
    print("left:{0}, right:{1} \n".format(len(masked_image[left_angle_mask]), len(masked_image[right_angle_mask])))
    left = np.mean(masked_image[left_angle_mask])
    right = np.mean(masked_image[right_angle_mask])
    return left, right

def plot_difference_of_angles(polar_image1, polar_image2):
    diff = polar_image2 - polar_image1
    angs, rads = get_polar_coordinates(diff)
    diffswaped = np.swapaxes(diff, 0, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='polar')
    cs = ax1.contourf(angs, rads, diffswaped[:, ::-1], cmap='RdBu')
    fig1.colorbar(cs)


#%%calculate stuff

polar_image = cv.linearPolar(source[0], center, max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_NEAREST)
polar_image = np.empty_like(source)
halfspace_intensity = np.empty((len(files), 2))
for i, s in enumerate(source):
     polar_image[i] = cv.linearPolar(s, center, max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_NEAREST)
     halfspace_intensity[i] = integrate_intensity_of_half_spaces(polar_image[i], 0.109, np.pi / 6,  272, 282)


plt.plot(image_angle, halfspace_intensity[:, 0])
plt.plot(image_angle, halfspace_intensity[:, 1])
plt.show()

# for i, pi in enumerate(polar_image):
#     plot_difference_of_angles(polar_image[i], polar_image[i+9])
#     plt.savefig("results/diff"+str(i)+'-'+str(i+9) +".png", dpi=300)









