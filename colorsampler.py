import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from picamera import PiCamera
from colorsys import rgb_to_hsv, rgb_to_hls

def rgb_to_cmyk(r, g, b):
    c = 1 - r
    m = 1 - g
    y = 1 - b
    k = min(c, m, y)
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return c, m, y, k

def rgb_to_lab(r, g, b):
    # Convert RGB to XYZ
    var_R = r / 255.0
    var_G = g / 255.0
    var_B = b / 255.0

    if var_R > 0.04045:
        var_R = ((var_R + 0.055) / 1.055) ** 2.4
    else:
        var_R = var_R / 12.92
    if var_G > 0.04045:
        var_G = ((var_G + 0.055) / 1.055) ** 2.4
    else:
        var_G = var_G / 12.92
    if var_B > 0.04045:
        var_B = ((var_B + 0.055) / 1.055) ** 2.4
    else:
        var_B = var_B / 12.92

    var_R *= 100
    var_G *= 100
    var_B *= 100

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    # Convert XYZ to LAB
    ref_X = 95.047
    ref_Y = 100.000
    ref_Z = 108.883

    var_X = X / ref_X
    var_Y = Y / ref_Y
    var_Z = Z / ref_Z

    if var_X > 0.008856:
        var_X **= (1/3)
    else:
        var_X = (7.787 * var_X) + (16/116)
    if var_Y > 0.008856:
        var_Y **= (1/3)
    else:
        var_Y = (7.787 * var_Y) + (16/116)
    if var_Z > 0.008856:
        var_Z **= (1/3)
    else:
        var_Z = (7.787 * var_Z) + (16/116)

    L = (116 * var_Y) - 16
    a = 500 * (var_X - var_Y)
    b = 200 * (var_Y - var_Z)

    return L, a, b

# Capture an image using PiCamera
camera = PiCamera()
camera.capture('image.jpg')

# Load the image using matplotlib.image
image = mpimg.imread('image.jpg')

# Get the center pixel coordinates
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2

# Get the color sample from the center of the image
color_sample_rgb = image[center_y-5:center_y+5, center_x-5:center_x+5]

# Convert the color sample to different color spaces
color_sample_hsv = np.array([rgb_to_hsv(*pixel) for pixel in color_sample_rgb.reshape(-1,3)]).reshape(color_sample_rgb.shape)
color_sample_hls = np.array([rgb_to_hls(*pixel) for pixel in color_sample_rgb.reshape(-1,3)]).reshape(color_sample_rgb.shape)
color_sample_cmyk = np.array([rgb_to_cmyk(*pixel) for pixel in color_sample_rgb.reshape(-1,3)]).reshape(color_sample_rgb.shape[0], color_sample_rgb.shape[1], 4)
color_sample_lab = np.array([rgb_to_lab(*pixel) for pixel in color_sample_rgb.reshape(-1,3)]).reshape(color_sample_rgb.shape)

# Calculate the color histograms for the sample region
color_hist_rgb, _ = np.histogramdd(color_sample_rgb.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)))
color_hist_hsv, _ = np.histogramdd(color_sample_hsv.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)))
color_hist_hls, _ = np.histogramdd(color_sample_hls.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)))
color_hist_cmyk, _ = np.histogramdd(color_sample_cmyk.reshape(-1, 4), bins=(8, 8, 8, 8), range=((0, 1), (0, 1), (0, 1), (0, 1)))
color_hist_lab, _ = np.histogramdd(color_sample_lab.reshape(-1, 3), bins=(8, 8, 8))

# Normalize the histograms
color_hist_rgb /= np.sum(color_hist_rgb)
color_hist_hsv /= np.sum(color_hist_hsv)
color_hist_hls /= np.sum(color_hist_hls)
color_hist_cmyk /= np.sum(color_hist_cmyk)
color_hist_lab /= np.sum(color_hist_lab)

# Display the color spectrograms
plt.subplot(231)
plt.imshow(color_hist_rgb)
plt.title('RGB')

plt.subplot(232)
plt.imshow(color_hist_hsv)
plt.title('HSV')

plt.subplot(233)
plt.imshow(color_hist_hls)
plt.title('HLS')

plt.subplot(234)
plt.imshow(color_hist_cmyk[:,:,0,:])
plt.title('CMYK')

plt.subplot(235)
plt.imshow(color_hist_lab)
plt.title('LAB')

plt.show()
