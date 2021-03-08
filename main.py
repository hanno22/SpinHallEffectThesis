import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import geometric_transform
import numpy as np
from matplotlib.pyplot import figure
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system/3806851
img = [
    rgb2gray(mpimg.imread('data/bfp_bb.bmp')),
    #mpimg.imread('data/bfp.bmp'),
    rgb2gray(mpimg.imread('data/fp_bb_dark.bmp'))
]

for im in img:
    plt.imshow(im)# ,cmap="binary")
    plt.colorbar()
    plt.show()


def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order.
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal
    # from a corner to the mid-point of img.
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)

#profile_data = radial_profile(img[0], [len(img[0][0])/2, len(img[0][1])/2])


#plt.plot(profile_data)
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
pol, (rads, angs) = topolar(img[0], order=5)
cs = ax.contourf(angs, rads, pol, cmap='plasma')
fig.colorbar(cs)
plt.show()

radial_profile = np.average(pol, axis=1, weights= pol >0)
plt.plot(radial_profile)
plt.show()