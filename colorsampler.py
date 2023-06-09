import cv2
import numpy as np
from libcamera import stills

# Capture an image using libcamera
camera = stills.CameraStill()
output = camera.capture(encoding='bgr')
image = np.frombuffer(output, dtype=np.uint8).reshape((camera.resolution[1], camera.resolution[0], 3))

# Get the center pixel coordinates
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2

# Get the color sample from the center of the image
color_sample = image[center_y-5:center_y+5, center_x-5:center_x+5]

# Calculate the color histogram for the sample region
color_hist = cv2.calcHist([color_sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# Normalize the histogram
cv2.normalize(color_hist, color_hist)

# Display the color spectrogram
cv2.imshow('Color Spectrogram', color_hist)
cv2.waitKey(0)
