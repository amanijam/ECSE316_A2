from PIL import Image
from numpy import asarray

import matplotlib.pyplot as plt

iFile = 'moonlanding.png'
# # load image as pixel array
# image = plt.imread(iFile)
# # summarize shape of the pixel array
# print(image.dtype)
# print(image.shape)
# # display the array of pixels as an image
# plt.imshow(image)
# plt.show()


# load the image
image = Image.open(iFile)
# convert image to numpy array
data = asarray(image)
print(type(data))
# summarize shape
print(data.shape)

# create Pillow image
image2 = Image.fromarray(data)
print(type(image2))

# summarize image details
print(image2.mode)
print(image2.size)

print(data)