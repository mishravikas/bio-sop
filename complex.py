import scipy
from scipy import ndimage

# read image into numpy array
# $ wget http://pythonvision.org/media/files/images/dna.jpeg
dna = scipy.misc.imread('dna.jpeg') # gray-scale image


# smooth the image (to remove small objects); set the threshold
dnaf = ndimage.gaussian_filter(dna, 16)
T = 25 # set threshold by hand to avoid installing `mahotas` or
       # `scipy.stsci.image` dependencies that have threshold() functions

# find connected components
labeled, nr_objects = ndimage.label(dnaf > T) # `dna[:,:,0]>T` for red-dot case
print "Number of objects is %d " % nr_objects

# show labeled image
####scipy.misc.imsave('labeled_dna.png', labeled)
####scipy.misc.imshow(labeled) # black&white image
import matplotlib.pyplot as plt
plt.imsave('im1.png', labeled)
plt.imshow(labeled)

plt.show()
