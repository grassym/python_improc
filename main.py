import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc
import numpy as np
import colormodels as colors

def LOG(text):
	print('\t...  '+ text + '  ...')


if __name__ == '__main__':
	filename = '/input.jpg' 
	image = misc.imread(filename)
	h, w = image.shape[0],image.shape[1]
	LOG("image opened")	
	
	ycrcb = colors.new_RGBtoYCrCb(image)
	[y,cr,cb] = np.dsplit(ycrcb,3)
	y = y.reshape(h,w)
	cr = cr.reshape(h,w)
	cb = cb.reshape(h,w)
	LOG("splitted to YCrCb channels")

	ycrcb = np.dstack((y,cr,cb))
	image = colors.new_YCrCbtoRGB(ycrcb)
	misc.imsave('/output.png',image)
	LOG("saved merged image")
