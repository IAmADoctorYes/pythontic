import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from skimage import color
import os

# Capture an image using libcamera-still
os.system('libcamera-still -o image.jpg')

# Load the image using matplotlib.image
image = mpimg.imread('image.jpg')

# Get the center pixel coordinates
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2

# Get the color sample from the center of the image
color_sample = image[center_y-50:center_y+50, center_x-50:center_x+50]

# Convert the color sample to the LAB color space
color_sample_lab = color.rgb2lab(color_sample)

# Calculate the average LAB color values
avg_lab = np.mean(color_sample_lab, axis=(0, 1))

# Display the average LAB color values
print(f'Average LAB color values: L={avg_lab[0]:.2f}, a={avg_lab[1]:.2f}, b={avg_lab[2]:.2f}')
