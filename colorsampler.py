import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Rectangle
import os

def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

# Capture an image using libcamera-still
os.system('libcamera-still -t 1000 -n --Shutter 8000 --awbgains 3.5,1.5 --denoise off -o image.jpg')

# Load the image using matplotlib.image
image = mpimg.imread('image.jpg')

# Convert the image from RGB to BGR
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
sample_size = 100
# Get the center pixel coordinates
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2

# Get the color sample from the center of the image
color_sample = image[center_y-sample_size:center_y+sample_size, center_x-sample_size:center_x+sample_size]

# Calculate the color histogram for the sample region
color_hist = cv2.calcHist([color_sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# Normalize the histogram
cv2.normalize(color_hist, color_hist)

# Create a 2D projection of the color histogram by summing along the first axis
color_hist_2d = np.sum(color_hist, axis=0)
average_R = 0
average_G = 0
average_B = 0
for column in color_sample:
    average_R += np.average([i[0] for i in column])
    average_G += np.average([i[1] for i in column])
    average_B += np.average([i[2] for i in column])
average_R /= len(color_sample)
average_G /= len(color_sample)
average_B /= len(color_sample)
print("RGB VALUES: ",average_R,average_G,average_B)
lab_color = rgb2lab([average_R,average_G,average_B])
print("LAB VALUES: ", lab_color[0],lab_color[1],lab_color[2])
# Display the color spectrogram using matplotlib.pyplot
plt.imshow(image)
plt.gca().add_patch(Rectangle((center_x,center_y),sample_size,sample_size, linewidth=1,edgecolor='b',facecolor='none'))
plt.show()

