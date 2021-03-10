import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import geometric_transform
import numpy as np
import cv2 as cv


img = cv.imread('data/2/bfp_bb.bmp', 0)
#img = cv.medianBlur(img, 15)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, param1=30, param2=20, minRadius=410, maxRadius=430)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

# cv.imshow('detected circles',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# probe_number = '1'
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
# img = [
#     rgb2gray(mpimg.imread('data/' + probe_number + '/bfp_bb.bmp')),
#     #mpimg.imread('data/bfp.bmp'),
#     #rgb2gray(mpimg.imread('/data2/fp_bb_dark.bmp'))
# ]

def topolar(data, center, order=1):
    max_radius = 0.5*np.linalg.norm(data.shape)

    def transform(coords):
        theta = 2*np.pi*coords[1] / (data.shape[1]) #- 1.)
        radius = max_radius * coords[0] / data.shape[0]

        i = data.shape[0] - radius*np.sin(theta) - center[0]
        j = radius*np.cos(theta) + data.shape[1] - center[1]
        return i, j

    polar = geometric_transform(data, transform, order=order)
    rads = max_radius * np.linspace(0, 1, data.shape[0])
    angs = np.linspace(0, 2*np.pi, data.shape[1])

    return polar, (rads, angs)

pol, (rads, angs) = topolar(img, center= (100,100), order=5)
plt.imshow(img)
plt.show()
null_indices = np.where(~pol.any(axis=1))[0]
pol = np.delete(pol, null_indices, 0)
rads = np.delete(rads, null_indices)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='polar')
NA = 1.216
radial_profile = np.average(pol, axis=1, weights= pol >0)
index_max = np.argmax(radial_profile)
index_NA = 710
radial_axis = np.arange(0, NA/index_NA * len(rads), NA/index_NA)

cs = ax1.contourf(angs, radial_axis, pol, cmap='binary')
fig1.colorbar(cs)
#fig1.safefig('results/' + probe_number + '/polarplot.png')
plt.show()




extra_ticks = [radial_axis[index_max], radial_axis[index_NA]]
fig2, ax2 = plt.subplots()
ax2.plot(radial_axis, radial_profile, '.')
ax2.axvline(radial_axis[index_max], linestyle='--', linewidth=0.5)
ax2.axvline(radial_axis[index_NA], linestyle='--', linewidth=0.5)
#ax2.set_xticks(list(ax2.get_xticks()) + extra_ticks)
print(radial_axis[index_max])
plt.show()