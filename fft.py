import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Default values
mode = 1
image = 'moonlanding.png'

# MODE
# [1] (Default) for fast mode where the image is converted into its FFT form and displayed
# [2] for denoising where the image is denoised by applying an FFT, truncating high 
#       frequencies and then displayed
# [3] for compressing and saving the image
# [4] for plotting the runtime graphs for the report

expSyntax = "fft.py [-m mode] [-i image]"
syntaxError = 0

# Parse input with expected format: python fft.py [-m mode] [-i image]
if len(sys.argv) > 5: 
    print("ERROR\tToo many args - Expected Syntax: {}".format(expSyntax))
    exit(1)
elif len(sys.argv) == 1:
    print("No args")
elif len(sys.argv) == 3:
    print("2 args (either mode or image)")
    if(sys.argv[1] == '-m'):
        mode = sys.argv[2]
    elif(sys.argv[1] == '-i'):
        image = sys.argv[2]
    else:
        syntaxError = 1
elif len(sys.argv) == 5:
    print("4 args (mode and image)")
    if(sys.argv[1] == '-m' and sys.argv[3] == '-i'):
        mode = sys.argv[2]
        image = sys.argv[4]
    else:
        syntaxError = 1
else:
    syntaxError = 1

if(syntaxError):
    print("ERROR\tIncorrect Syntax - Expected Syntax: {}".format(expSyntax))
    exit(1)

def naiveDFT(arr):
    arr = np.asarray(arr, dtype=complex)
    transform = arr.copy()
    for i in range(len(arr)):
        transform[i] = np.sum(arr*np.exp((-2j*np.pi)/len(arr)*i*np.arange(len(arr))))

    return np.asarray(transform)

def inverseNaive(arr):
    arr = np.asarray(arr, dtype=complex)
    transform = arr.copy()
    for i in range(len(arr)):
        transform[i] = np.sum(arr*np.exp((2j*np.pi)/len(arr)*i*np.arange(len(arr))))/len(arr)

    return np.asarray(transform)

def naive2d(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = naiveDFT(row)
    
    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = naiveDFT(col)

    arr = arr.transpose()

    return arr

def inverseNaive2d(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = inverseNaive(row)
    
    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = inverseNaive(col)

    arr = arr.transpose()

    return arr

def denoiseMode(img):
    img = plt.imread(img)

    denoised_img = naive2d(img)
    for i, row in enumerate(denoised_img):
        for j, col in enumerate(row):
            if col.real > (160*np.pi):
                denoised_img[i][j] = 0
    denoised_img = inverseNaive2d(denoised_img)

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img.real)

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_img.real)
    plt.show()

if mode == '2':
    denoiseMode(image)


# if __name__ == "__main__":

#     x = [[2,3,4], [4,3,5], [7,5,6]]

#     print(naive2d(x))
#     print(np.fft.fft2(x))
#     print("hello")
#     img = plt.imread(image).astype(float)
#     # img = cv2.imread(image)
#     print("hello")
#     ours = naive2d(img)
#     print("hello")
#     plt.imshow(ours)
#     plt.show()
#     test_result = np.fft.fft2(img)
#     print(test_result)
#     print(naive2d(img))
#     # print(np.fft.fft2(img))