# %%import libaries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import geometric_transform
import numpy as np
import cv2 as cv

# %%load data
center = (1238, 1011)
max_radius = 500
source = cv.imread('data/spin_hall/1/0.bmp')
data = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
#%% load test data
center = (213, 206)
max_radius = 300
source = cv.imread('data/spin_hall/polar_remap.png')
data = cv.cvtColor(source, cv.COLOR_BGR2GRAY)


#%% plotpolar
def plot_polar(polar_image):
    polar_image = np.swapaxes(polar_image, 0, 1)
    rads = np.linspace(0, max_radius, polar_image.shape[0])
    angs = np.linspace(0, 2 * np.pi, polar_image.shape[1])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='polar')
    cs = ax1.contourf(angs, rads, polar_image[:, ::-1], cmap='binary')
    fig1.colorbar(cs)
    plt.savefig('test.png', dpi=500)


#%%calculate stuff
polar_image = cv.linearPolar(data, center, max_radius, cv.WARP_FILL_OUTLIERS, cv.INTER_CUBIC)
plot_polar(polar_image)

