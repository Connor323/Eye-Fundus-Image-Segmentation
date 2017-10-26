
# This script is to generate mathched filter response with both Gaussian function and 
# first-order derivatie of Gaussian function. 

# The input files are original color image and its mark image. The out put files are 
# matched filter response with Gaussian filter (grayscale image, H+name.tiff) and the image 
# threshold with matched filter response with first-order derivatie of Gaussian filter (binary 
# image out+name.tiff).

# There are 4 parameters to be adjust, L: the length of the neighborhood along the y-axis, 
# sigma: the standard deviation of Gaussian function, w: the kernel size of the low-pass 
# filter before compute MFR-FDoG, c: threshold coefficient.

import numpy as np
import cv2
from math import exp, pi, sqrt
from numbapro import vectorize
import sys

class MFR:
    '''
    MFR class requests L, sigme, w, c to inicialize. 

    This class is used for generate filter of Gaussian (gaussian_matched_filter_kernel) and 
    first-order derivative of Gaussian (fdog_filter_kernel)and the filter bank of the previous 
    filters (createMatchedFilterBank). applyFilters function used for generate the matched filter
    response of the given filter. 

    '''

    def __init__(self, L, sigma, w, c):
        if L == 0:
            L = 1
        if sigma == 0:
            sigma = 1
        if w == 0:
            w = 1
        self.L = L
        self.sigma = sigma
        self.w = w
        self.c = c

    def _filter_kernel_mf_fdog(self, t = 3, mf = True):
        dim_y = int(self.L)
        dim_x = int(2 * t * self.sigma)
        if dim_x == 0:
            dim_x = 1
        if dim_y == 0:
            dim_y = 1
        arr = np.zeros((dim_y, dim_x), 'f')
        
        ctr_x = dim_x / 2 
        ctr_y = int(dim_y / 2.)

        # an un-natural way to set elements of the array
        # to their x coordinate. 
        # x's are actually columns, so the first dimension of the iterator is used
        it = np.nditer(arr, flags=['multi_index'])
        while not it.finished:
            arr[it.multi_index] = it.multi_index[1] - ctr_x
            it.iternext()

        two_sigma_sq = 2 * self.sigma * self.sigma
        if two_sigma_sq == 0:
            two_sigma_sq = 1
        div = sqrt(2 * pi) * self.sigma
        if div == 0:
            div = 1.
        sqrt_w_pi_sigma = 1. / div
        if not mf:
            div = self.sigma ** 2
            if div == 0:
                div = 1.
            sqrt_w_pi_sigma = sqrt_w_pi_sigma / div

        @vectorize(['float32(float32)'], target='cpu')
        def k_fun(x):
            return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

        @vectorize(['float32(float32)'], target='cpu')
        def k_fun_derivative(x):
            return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

        if mf:
            kernel = k_fun(arr)
            kernel = kernel - np.sum(kernel)/(dim_x*dim_y)
        else:
           kernel = k_fun_derivative(arr)

        # return the "convolution" kernel for filter2D
        return kernel

    def fdog_filter_kernel(self, t = 3):
        '''
        K = - (x/(sqrt(2 * pi) * sigma ^3)) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
        '''
        return self._filter_kernel_mf_fdog(t, False)

    def gaussian_matched_filter_kernel(self, t = 3):
        '''
        K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
        '''
        return self._filter_kernel_mf_fdog(t, True)

    def createMatchedFilterBank(self, K, n = 12):
        '''
        Given a kernel, create matched filter bank
        '''
        rotate = 180 / n
        center = (K.shape[1] / 2, K.shape[0] / 2)
        cur_rot = 0
        kernels = [K]

        for i in range(1, n):
            cur_rot += rotate
            r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
            k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
            if np.count_nonzero(k):
                mean = np.sum(k)/np.count_nonzero(k)
            else: 
                mean = 0
            for y in range(len(k)):
                for x in range(len(k[0])):
                    if k[y][x]:
                        k[y][x] -= mean
            kernels.append(k)

        return kernels

    #Applying the filter bank
    def applyFilters(self, im, kernels):
        '''
        Given a filter bank, apply them and record maximum response
        '''
        images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
        return np.max(images, 0)

#######Parameters for DRIVE##################################
# L = 5      # the length of the neighborhood along the y-axis to smooth noise
# sigma = 1
# w = 31    # kernel size
# c = 1 # the gain of threshold
#######Parameters for STARE##################################
L = 9      # the length of the neighborhood along the y-axis to smooth noise
sigma = 1.5
w = 31    # kernel size
c = 1.5 # the gain of threshold

def inbounds(shape, indices):
    '''
    Test if the given coordinates inside the given image. 

    The first input parameter is the shape of image (height, weight) and the 
    second parameter is the coordinates to be tested (y, x)

    The function returns True if the coordinates inside the image and vice versa.

    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True

def setlable(img, labimg, x, y, label, size):
    '''
    This fucntion is used for label image. 

    The first two input images are the image to be labeled and an output image with 
    labeled region. "x", "y" are the coordinate to be tested, "label" is the ID
    of a region and size is used to limit maximum size of a region. 

    '''
    if img[y][x] and not labimg[y][x]:
        labimg[y][x] = label
        size += 1
        if size > 500:
                return False
        if inbounds(img.shape, (y, x+1)):
            setlable(img, labimg, x+1, y,label, size)
        if inbounds(img.shape, (y+1, x)):
            setlable(img, labimg, x, y+1,label, size)
        if inbounds(img.shape, (y, x-1)):
            setlable(img, labimg, x-1, y,label, size)
        if inbounds(img.shape, (y-1, x)):
            setlable(img, labimg, x, y-1,label, size)
        if inbounds(img.shape, (y+1, x+1)):
            setlable(img, labimg, x+1, y+1,label, size)
        if inbounds(img.shape, (y+1, x-1)):
            setlable(img, labimg, x-1, y+1,label, size)
        if inbounds(img.shape, (y-1, x+1)):
            setlable(img, labimg, x+1, y-1,label, size)
        if inbounds(img.shape, (y-1, x-1)):
            setlable(img, labimg, x-1, y-1,label, size)

# im0 is the original image and mask is the mask image of the given image
im0 = cv2.imread(sys.argv[1])
mask = cv2.imread(sys.argv[2])

# we split the orignal image into 3 channels, b, g, r and only use green 
# channel. Also, we convert the mask image into a grayscale image.
b,g,r = cv2.split(im0) 
im1 = g
h, w = im1.shape[:2]
im1 = 255 - im1
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# initialize a variable of the class of MFR. 
matched = MFR(L, sigma, w, c)

# generate Gaussian filter and first-order derivative of Gaussian filter.
gf = matched.gaussian_matched_filter_kernel()
fdog = matched.fdog_filter_kernel()

# generate filter bank
bank_gf = matched.createMatchedFilterBank(gf, 12)
bank_fdog = matched.createMatchedFilterBank(fdog, 12)

# obtain matched filter response. H is the MFR-G and D is MFR-FDoG
H = matched.applyFilters(im1, bank_gf)
D = matched.applyFilters(im1, bank_fdog)

# compute the threshold value using MFR-FDoG
kernel_size = 31
kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)
dm = np.zeros(D.shape,np.float32)
DD = np.array(D, dtype='f')
dm = cv2.filter2D(DD,-1,kernel)
dmn = cv2.normalize(dm, 0, 1, cv2.NORM_MINMAX)
uH = cv2.mean(H)
Tc = matched.c * uH[0]
T = (1+dmn) * Tc 

# threshold the MFR-G with previous threhshold value.
out = np.zeros(H.shape)
out[H > T] = 255

# using the mask image to truncate the value outside the reina. 
laplacian = cv2.Laplacian(mask, cv2.CV_64F)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
laplacian = cv2.dilate(laplacian,kernel,iterations = 4)
H[(mask == 0) + (laplacian != 0)] = 0
out[(mask == 0) + (laplacian != 0)] = 0

# get rid of the segment less than 10 pixel 
lab = 1
label = np.zeros(out.shape)
for y in range(h):
    for x in range(w):
        if not label[y][x] and out[y][x]:
            size = 0
            setlable(out, label, x, y, lab, size)
            lab += 1
num = np.zeros(lab)
for y in range(h):
    for x in range(w):
        num[label[y][x]-1] += 1
for y in range(h):
    for x in range(w):
        if num[label[y][x]-1] <= 10:
            out[y][x] = 0

# generate the output images. 
cv2.imwrite("Final_" + sys.argv[1], out)
cv2.imwrite("MF_" + sys.argv[1], H)
