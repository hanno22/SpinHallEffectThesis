# %%import libaries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import geometric_transform
import numpy as np
import cv2 as cv

# %%load data
center = (1238, 1011)
max_radius = 500
source = cv.imread('data/spin_hall/1/40.bmp')
data = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
#%% load test data
center = (213, 206)
max_radius = 300
source = cv.imread('data/spin_hall/polar_remap.png')
data = cv.cvtColor(source, cv.COLOR_BGR2GRAY)

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
    fig1.colorbar(cs)
    plt.savefig('test.png', dpi=500)


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
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plt.savefig('test.png', dpi=500)

def integrate_intensity_of_half_spaces(polar_image, angle, min_rad, max_rad):
    angs, rads = get_polar_coordinates(polar_image)
    mask = (angs > angle) & (angs < angle + np.pi)

    left = polar_image[mask, :]
    print(left)








#%%calculate stuff
polar_image = cv.linearPolar(data, center, max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_CUBIC)
plot_polar(polar_image)


