# %%import libaries
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import os
import re

pol_data = np.genfromtxt('data/polorientation.csv', delimiter=',')

center = [(1238, 1011), (1201, 1128), (1203, 1130)]
datasetnumber = 2


# center = (563, 338)

max_radius = 500
# %%load data
path = 'data/spin_hall/{0}'.format(datasetnumber)
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



# %% functions
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


def integrate_intensity_of_half_spaces(polar_image, mid_angle, angle_width, min_rad, max_rad, remove_angs = 0):
    angs, rads = get_polar_coordinates(polar_image)
    left_angle_mask, right_angle_mask = get_angle_masks(mid_angle, angle_width, angs, remove_angs)
    radial_mask = (rads > min_rad) & (rads < max_rad)
    masked_image = polar_image[:, radial_mask]
    #print("left:{0}, right:{1} \n".format(len(masked_image[left_angle_mask]), len(masked_image[right_angle_mask])))
    left_right = np.sum(masked_image[left_angle_mask | right_angle_mask])
    left = np.mean(masked_image[left_angle_mask]) / left_right
    right = np.mean(masked_image[right_angle_mask]) / left_right
    return left, right


def get_angle_masks(mid_angle, angle_width, angs, remove_angs = 0):
    if mid_angle + angle_width < 2 * np.pi:
        if mid_angle + remove_angs < 2 * np.pi:
            left_angle_mask = (angs >= mid_angle + remove_angs) & (angs <= mid_angle + angle_width)
        else:
            raise Exception('something Wrong: mid_angle + remove_angs > 2 * np.pi:')
    else:
        if mid_angle + remove_angs < 2 * np.pi:
            left_angle_mask = (angs >= mid_angle + remove_angs) | (angs <= mid_angle + angle_width - (2 * np.pi))
        else:
            left_angle_mask = (angs >= mid_angle + remove_angs - 2 * np.pi) & (angs <= mid_angle + angle_width - (2 * np.pi))

    if mid_angle - angle_width > 0:
        if(mid_angle - remove_angs) > 0:
            right_angle_mask = (angs <= mid_angle - remove_angs) & (angs >= mid_angle - angle_width)
        else:
            raise Exception('(mid_angle - remove_angs) < 0')
    else:
        if (mid_angle - remove_angs) > 0:
            right_angle_mask = (angs <= mid_angle - remove_angs) | (angs >= 2 * np.pi + (mid_angle - angle_width))
        else:
            right_angle_mask = (angs <= mid_angle - remove_angs) |\
                               ((angs >= 2 * np.pi + (mid_angle - angle_width)) & (angs <= 2 * np.pi + mid_angle - remove_angs))
    return left_angle_mask, right_angle_mask

def plot_masks(polar_image, mid_angle, angle_width, min_rad, max_rad, remove_angs = 0):
    angs, rads = get_polar_coordinates(polar_image)
    polar_image_swaped = np.swapaxes(polar_image, 0, 1)
    left_mask, right_mask = get_angle_masks(mid_angle, angle_width, angs, remove_angs)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='polar')
    cs = ax1.contourf(angs, rads, polar_image_swaped[:, ::-1], cmap='binary')
    left_mask= sorted(angs[left_mask])
    right_mask = sorted(angs[right_mask])
    ax1.plot(right_mask, np.full(len(right_mask), min_rad), '-.g', linewidth=0.5)
    ax1.plot(right_mask, np.full(len(right_mask), max_rad), '-.g', linewidth=0.5)

    ax1.plot(left_mask, np.full(len(left_mask), min_rad), '-.r', linewidth=0.5)
    ax1.plot(left_mask, np.full(len(left_mask), max_rad), '-.r', linewidth=0.5)


    #ax1.plot(angs, np.full(angs.shape, 450), linewidth=0.3)
    ax1.plot(np.full(rads.shape, mid_angle), rads, linewidth=0.4)
    #fig1.colorbar(cs)
    plt.show()

def plot_difference_of_angles(polar_images, i, j):
    diff = (polar_images[j] - polar_images[i]) / (polar_images[i] + polar_images[j])
    angs, rads = get_polar_coordinates(diff)
    diffswaped = np.swapaxes(diff, 0, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='polar')
    cs = ax1.contourf(angs, rads, diffswaped[:, ::-1], cmap='RdBu')
    fig1.colorbar(cs)
    fig1.suptitle('Diff {0}-{1}'.format(image_angle[i], image_angle[j]))
    plt.show()


# %%calculate stuff
spin_hall_angle = 0#0.15
angle_width = np.pi / 2
remove_angs = np.pi / 8
min_rad = 425
max_rad = 448
polar_images = np.empty_like(source)
halfspace_intensity = np.empty((len(files), 2))
for i, s in enumerate(source):
    polar_images[i] = cv.linearPolar(s, center[datasetnumber], max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_NEAREST)
    #halfspace_intensity[i] = integrate_intensity_of_half_spaces(polar_image[i], spin_hall_angle, angle_width, min_rad, max_rad)
#polar_image = polar_image - polar_image[13]
for i, pi in enumerate(polar_images):
    halfspace_intensity[i] = integrate_intensity_of_half_spaces(polar_images[i], spin_hall_angle, angle_width, min_rad, max_rad, remove_angs)
plot_masks(polar_images[0], spin_hall_angle, angle_width, min_rad, max_rad, remove_angs)
plt.title("angle{0}".format(spin_hall_angle))
plt.plot(image_angle, halfspace_intensity[:, 0], 'g')
plt.plot(image_angle, halfspace_intensity[:, 1], 'r')
plt.show()

plot_difference_of_angles(polar_images, 24, 37)

# halfspace_angles = np.linspace(1.25, 1.6, 10)
# polar_image = np.empty_like(source)
# halfspace_intensity = np.empty((len(halfspace_angles), len(files), 2))
# for i, s in enumerate(source):
#     polar_image[i] = cv.linearPolar(s, center, max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_NEAREST)
#     for j, a in enumerate(halfspace_angles):
#         halfspace_intensity[j][i] = integrate_intensity_of_half_spaces(polar_image[i], a, np.pi / 6, 430, 440)
# #plot_masks(polar_image[5], 1.48, np.pi / 6, 425, 445)
# for i, hi in enumerate(halfspace_intensity):
#     plt.title("angle{0}".format(halfspace_angles[i]))
#     plt.plot(image_angle, hi[:, 0])
#     plt.plot(image_angle, hi[:, 1])
#     plt.show()

# for i, pi in enumerate(polar_image):
#     plot_difference_of_angles(polar_image[i], polar_image[i+9])
#     plt.savefig("results/diff"+str(i)+'-'+str(i+9) +".png", dpi=300)
