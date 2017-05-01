from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
# load the image and show it
image = cv2.imread(args["image"])
image2 = cv2.imread("im2.png")
image3 = cv2.imread("im3.png")
cv2.imshow("image2",image2)
cv2.imshow("image", image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
 
# loop over the image channels
for (chan, color) in zip(chans, colors):
	if color!="b":
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
for (chan, color) in zip(chans, colors):
	if color!="b":
		continue
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
 
	# plot the histogram
	plt.plot(hist, color = "r")
	plt.xlim([0, 256])

#For image3
chans=cv2.split(image3)
for (chan, color) in zip(chans, colors):
	if color!="b":
		continue
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
 
	# plot the histogram
	plt.plot(hist, color = "g")
	plt.xlim([0, 256])

 
# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print "flattened feature vector size: %d" % (np.array(features).flatten().shape)
plt.show()