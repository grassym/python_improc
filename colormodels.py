import __future__ 
import numpy as np

matr_to_ycrcb_mult = np.array([ [0.299, 0.587, 0.114], [0.5, -0.4187, -0.0813], [-0.1687, -0.3313, 0.5] ])
vec_to_ycrcb_add = np.array([ 0, 128, 128 ])
matr_to_rgb_mult = np.array([ [1,1.402,0], [1,-0.71414,-0.34414], [1,0,1.772] ])
vec_to_rgb_add = np.array([ -128*1.402, 128*1.05828, -128*1.772 ])

def imageRGBtoYCrCb(rgb_image):
	rgb_image = np.apply_along_axis(RGBtoYCrCb, 2, rgb_image)
	return rgb_image

def imageYCrCbtoRGB(ycrcb_image):
	ycrcb_image = np.apply_along_axis(YCrCbtoRGB, 2, ycrcb_image)
	return ycrcb_image

def RGBtoYCrCb(rgb):
	""" Converts RGB vector to YCrCb vector
	"""
	return np.dot(matr_to_ycrcb_mult,rgb) + vec_to_ycrcb_add

def YCrCbtoRGB(ycrcb):
	""" Converts YCrCb vector to RGB vector
	"""
	return np.dot(matr_to_rgb_mult,ycrcb) + vec_to_rgb_add

def new_RGBtoYCrCb(image):
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    y = 0.299*r + 0.587*g + 0.114*b
    cb = 128 - 0.1687*r - 0.3313*g + 0.5*b 
    cr = 128 + 0.5*r - 0.4187*g - 0.0813*b

    return np.dstack((y, cr, cb))

def new_YCrCbtoRGB(image):
    y, cr, cb = image[..., 0], image[..., 1], image[..., 2]

    r = 1.402*(cr-128) + y
    g = -0.34414*(cb-128) - 0.71414*(cr-128) + y
    b = 1.772*(cb-128) + y

    return np.dstack((r, g, b))