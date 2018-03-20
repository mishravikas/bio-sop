# bio-sop

**Algorithm for light and dark microscopic image analysis**

**Method 1 - Counting the no. of objects**

Software Requirements:

- Python 2.5, 2.6, or 2.7
- numpy
- matplotlib
- mahotas
- Ipython

**First Task: Counting the no of objects**

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image16.png)
Our first task will be to take this image and count the number of objects in it. We are going to threshold the image and count the number of objects.

T = mh.thresholding.otsu(img)
pylab.imshow(img &gt; T)
pylab.show()

Here, again, we are taking advantage of the fact that img is a numpy array and using it in logical operations (img &gt; T). The result is a numpy array of booleans, which pylab shows as a black and white image (or red and blue if you have not previously called pylab.gray()).

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image17.png)
This isn&#39;t too good. The image contains many small objects. There are a couple of ways to solve this. A simple one is to smooth the image a bit using a Gaussian filter.

imgf = mh.gaussian\_filter(imgf, 8)
T = mh.thresholding.otsu(imgf)
pylab.imshow(imgf &gt; T)
pylab.show()

The function mh.gaussian\_filter takes an image and the standard deviation of the filter (in pixel units) and returns the filtered image. We are jumping from one package to the next, calling mahotas to filter the image and to compute the threshold, using numpy operations to create a thresholded images, and pylab to display it, but everyone works with numpy arrays. The result is much better:

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image9.png)
We now have some merged objects (those that are touching), but overall the result looks much better. The final count is only one extra function call away:

labeled,nr\_objects = mh.label(imgf &gt; T)
print nr\_objects
pylab.imshow(labeled)
pylab.jet()
pylab.show()

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image1.png)
**Second Task: Segmenting the Image**

The previous result was acceptable for a first pass, but there were still objects glued together. Let&#39;s try to do better.

Here is a simple, traditional, idea:

- smooth the image
- find regional maxima (color intensity)
- Use the regional maxima as seeds for watershed

Finding the seeds:

imgf = mh.gaussian\_filter(img, 8)
rmax = mh.regmax(imgf)
pylab.imshow(mh.overlay(img, rmax))
pylab.show()

The mh.overlay() returns a colour image with the grey level component being given by its first argument while overlaying its second argument as a red channel. The result doesn&#39;t look so good:

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image15.png)
If we look at the filtered image, we can see the multiple maxima:

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image7.png)
After a little fiddling around, we decide to try the same idea with a bigger sigma value:

Taking sigma value as 16 instead of 8 the results look much better

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image14.png)
We can easily count the no of objects now

**Images being compared**

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image12.png)
                Image 1                                       
 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image11.png)
                Image 2                                         
 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image13.png)               
                Image 3

**Method 2**

Flattened color histogram

 A histogram represents the distribution of colors in an image. It can be visualized as a graph (or plot) that gives a high-level intuition of the intensity (pixel value) distribution. We are going to assume a RGB color space in this example, so these pixel values will be in the range of 0 to 255.

When plotting the histogram, the X-axis serves as our &quot;bins&quot;. If we construct a histogram with 256 bins, then we are effectively counting the number of times each pixel value occurs. In contrast, if we use only 2 (equally spaced) bins, then we are counting the number of times a pixel is in the range [0, 128) or [128, 255]. The number of pixels binned to the X-axis value is then plotted on the Y-axis.

**Algorithm/Source Code**

from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add\_argument(&quot;-i&quot;, &quot;--image&quot;, required =True, help =&quot;Path to the image&quot;)
args =vars(ap.parse\_args())

# load the image and show it
image = cv2.imread(args[&quot;image&quot;])
image2 = cv2.imread(&quot;im2.png&quot;)
image3 = cv2.imread(&quot;im3.png&quot;)
cv2.imshow(&quot;image2&quot;,image2)
cv2.imshow(&quot;image&quot;, image)

# gray = cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY)
# cv2.imshow(&quot;gray&quot;, gray)
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure()
# plt.title(&quot;Grayscale Histogram&quot;)
# plt.xlabel(&quot;Bins&quot;)
# plt.ylabel(&quot;# of Pixels&quot;)
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()

chans = cv2.split(image)
colors = (&quot;b&quot;, &quot;g&quot;, &quot;r&quot;)
plt.figure()
plt.title(&quot;&#39;Flattened&#39; Color Histogram&quot;)
plt.xlabel(&quot;Bins&quot;)
plt.ylabel(&quot;# of Pixels&quot;)
features = []

# loop over the image channels
for (chan, color) inzip(chans, colors):
    if color!=&quot;b&quot;:
        continue
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

#For image2

chans=cv2.split(image2)
for (chan, color) inzip(chans, colors):
    if color!=&quot;b&quot;:
        continue
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color =&quot;r&quot;)
    plt.xlim([0, 256])

#For image3
chans=cv2.split(image3)
for (chan, color) inzip(chans, colors):
    if color!=&quot;b&quot;:
        continue
    # create a histogram for the current channel and
    # concatenate the resulting histograms for each
    # channel
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)

    # plot the histogram
    plt.plot(hist, color =&quot;g&quot;)
    plt.xlim([0, 256])


# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print&quot;flattened feature vector size: %d&quot;% (np.array(features).flatten().shape)
plt.show()

 ![]![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image6.png)
Blue - Image 1   Red- Image 2   Green - Image3

What does this histogram tell us? Well, there is a peak in the blue pixel values around bin #30. The peaks are also different for different images, it is evident from the graph that third image has a higher peak hence Blue channel is abundant in the third image. Ranking the 3 images accordingly on the basis on no of pixels in the blue channel in the RGB space: Image 3&gt; Image2&gt;Image1

**Method 3**

HSV 3D plot

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image10.png)
Intensity plot for Image1

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image4.png)
Intensity Plot for Image2

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image5.png)
Intensity plot for Image 3

Algorithm/Source Code

**import**** numpy ****as**** np**
**import**** mpl\_toolkits.mplot3d.axes3d ****as**** p3**
**import**** matplotlib.pyplot ****as**** plt**
**import**** colorsys**
**from**** PIL ****import** Image

_# (1) Import the file to be analyzed!_
img\_file = Image.open(&quot;im3.png&quot;)
img = img\_file.load()

_# (2) Get image width &amp; height in pixels_
[xs, ys] = img\_file.size
max\_intensity =100
hues = {}

_# (3) Examine each pixel in the image file_
**for** x **in** xrange(0, xs):
   **for** y **in** xrange(0, ys):
    _# (4)  Get the RGB color of the pixel_
    [r, g, b] = img[x, y]

    _# (5)  Normalize pixel color values_
    r /=255.0
    g /=255.0
    b /=255.0

    _# (6)  Convert RGB color to HSV_
    [h, s, v] = colorsys.rgb\_to\_hsv(r, g, b)

    _# (7)  Marginalize s; count how many pixels have matching (h, v)_
     **if** h **not**** in** hues:
      hues[h] = {}
     **if** v **not**** in** hues[h]:
      hues[h][v] =1
     **else** :
       **if** hues[h][v] &lt; max\_intensity:
        hues[h][v] +=1

_# (8)   Decompose the hues object into a set of one dimensional arrays we can use with matplotlib_
h\_ = []
v\_ = []
i = []
colours = []

**for** h **in** hues:
   **for** v **in** hues[h]:
    h\_.append(h)
    v\_.append(v)
    i.append(hues[h][v])
    [r, g, b] = colorsys.hsv\_to\_rgb(h, 1, v)
    colours.append([r, g, b])

_# (9)   Plot the graph!_
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter(h\_, v\_, i, s=5, c=colours, lw=0)

ax.set\_xlabel(&#39;Hue&#39;)
ax.set\_ylabel(&#39;Value&#39;)
ax.set\_zlabel(&#39;Intensity&#39;)
fig.add\_axes(ax)
plt.show()

**Method 4**

K-Means Color Clustering

This method helps in finding the dominant colors (i.e. the colors that are represented most in the image) in an image and also find the ratio of colors in that image. Since the three images to be tested are all of same size the ratios of the blue color can be used to quantify that color.

 ![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image8.png)
      Image1
![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image2.png)
      Image 2

![](https://raw.githubusercontent.com/mishravikas/bio-sop/master/image3.png)
      Image3

Algorithm/Source Code

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add\_argument(&quot;-i&quot;, &quot;--image&quot;, required =True, help =&quot;Path to the image&quot;)
ap.add\_argument(&quot;-c&quot;, &quot;--clusters&quot;, required =True, type =int,
    help =&quot;# of clusters&quot;)
args =vars(ap.parse\_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args[&quot;image&quot;])
image = cv2.cvtColor(image, cv2.COLOR\_BGR2RGB)

# show our image
plt.figure()
plt.axis(&quot;off&quot;)
plt.imshow(image)


# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] \* image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n\_clusters = args[&quot;clusters&quot;])
clt.fit(image)


# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid\_histogram(clt)
bar = utils.plot\_colors(hist, clt.cluster\_centers\_)

# show our color bart
plt.figure()
plt.axis(&quot;off&quot;)
plt.imshow(bar)
plt.show()
