import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timeit
import math

# Default values
mode = 1
iFile = "moonlanding.png"

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

    num_rows = len(img)
    num_cols = len(img[0])

    fraction = 0.075

    denoised_img = naive2d(img)
    denoised_img[round(fraction*num_rows):round((1-fraction)*num_rows)] = 0
    denoised_img[:, round(fraction*num_cols):round((1-fraction)*num_cols)] = 0
    denoised_img = inverseNaive2d(denoised_img)

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img.real, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_img.real, cmap="gray")
    plt.show()

def plottingMode():

    size_list = []
    naive_list = []
    fft_list = []
    for size_index in range(6,12,2):
        prob_size = 2**size_index
        size_list.append(prob_size)
        dimension = int(math.sqrt(prob_size))
        rand_values = np.random.random((dimension, dimension))
        naive_runs = []
        fft_runs = []
        for run_index in range(10):

            naive_start = timeit.timeit()
            naive2d(rand_values)
            naive_end = timeit.timeit()
            naive_time = naive_end - naive_start
            naive_runs.append(naive_time)

    return 0


if __name__ == "__main__":

#     x = [[2,3,4], [4,3,5], [7,5,6]]

    img = cv2.imread(iFile, cv2.IMREAD_GRAYSCALE)
    width = len(img[0])
    height = len(img)

    if int(np.log2(width)) != np.log2(width):
        width = pow(2, int(np.log2(width)) + 1)

    if int(np.log2(height)) != np.log2(height):
        height = pow(2, int(np.log2(height)) + 1)

    img = cv2.resize(img, (width, height))

    if mode == '2':
        denoiseMode(img)
    elif mode == '4':
        plottingMode()