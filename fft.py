import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import math

# MODE
# [1] (Default) for fast mode where the image is converted into its FFT form and displayed
# [2] for denoising where the image is denoised by applying an FFT, truncating high
#       frequencies and then displayed
# [3] for compressing and saving the image
# [4] for plotting the runtime graphs for the report

smallSize = 16


def naiveDFT(arr):
    arr = np.asarray(arr, dtype=complex)
    transform = arr.copy()
    for i in range(len(arr)):
        transform[i] = np.sum(
            arr * np.exp((-2j * np.pi) / len(arr) * i * np.arange(len(arr)))
        )

    return np.asarray(transform)


def inverseNaiveDFT(arr):
    arr = np.asarray(arr, dtype=complex)
    transform = arr.copy()
    for i in range(len(arr)):
        transform[i] = np.sum(
            arr * np.exp((2j * np.pi) / len(arr) * i * np.arange(len(arr)))
        )

    return np.asarray(transform)


def naive2d_DFT(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = naiveDFT(row)

    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = naiveDFT(col)

    arr = arr.transpose()

    return arr


def inverseNaive2d_DFT(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = inverseNaiveDFT(row)

    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = inverseNaiveDFT(col)

    arr = arr.transpose()

    return arr


def FFT(arr):
    n = len(arr)
    arr = np.asarray(arr, dtype=complex)
    if n < smallSize:
        return naiveDFT(arr)
    else:
        arr_even = FFT(arr[::2])
        arr_odd = FFT(arr[1::2])
        k = np.arange(n)
        exp = np.exp((-2j * np.pi * k) / n)
        arr_even = np.concatenate((arr_even, arr_even))
        arr_odd = np.concatenate((arr_odd, arr_odd))
        return arr_even + exp * arr_odd


def inverseFFT(arr):
    n = len(arr)
    arr = np.asarray(arr, dtype=complex)
    if n < smallSize:
        return inverseNaiveDFT(arr) * n
    else:
        arr_even = inverseFFT(arr[::2])
        arr_odd = inverseFFT(arr[1::2])
        k = np.arange(n)
        exp = np.exp((2j * np.pi * k) / n)
        arr_even = np.concatenate((arr_even, arr_even))
        arr_odd = np.concatenate((arr_odd, arr_odd))
        return (arr_even + exp * arr_odd) / n


def FFT2d(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = FFT(row)

    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = FFT(col)
    arr = arr.transpose()

    return arr


def inverseFFT2d(arr):
    arr = np.asarray(arr, dtype=complex)

    for row_index, row in enumerate(arr):
        arr[row_index] = inverseFFT(row)

    arr = arr.transpose()
    for col_index, col in enumerate(arr):
        arr[col_index] = inverseFFT(col)
    arr = arr.transpose()

    return arr


def fastMode(img):
    transfImg = FFT2d(img)

    plt.figure(figsize=(10, 5))  # 10x5 figure

    # first subplot of the original image (grayscale)
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    # second subplot of the fourier transform of the image, log scaled
    plt.subplot(1, 2, 2)
    plt.title("Fourier Transform")
    plt.imshow(np.abs(transfImg), norm=LogNorm(vmin=5))
    plt.colorbar()

    plt.show()

def denoiseMode(img):

    num_rows = len(img)
    num_cols = len(img[0])

    fraction = 0.09

    denoised_img = naive2d_DFT(img)
    denoised_img[round(fraction*num_rows):round(num_rows - (fraction*num_rows))] = (0 + 0j)
    denoised_img[:, round(fraction*num_cols):round(num_cols - (fraction*num_cols))] = (0 + 0j)
    denoised_img = inverseNaive2d_DFT(denoised_img)

    print("Number of non-zero rows: " + str(round(num_rows*fraction*2)))
    print("Number of non-zero columns: " + str(round(num_cols*fraction*2)))
    print("Fraction of non-zeroes: " + str(fraction)) 
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img.real, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_img.real, cmap="gray")
    plt.show()

def compressionMode(img):
    # 0% compression
    originalTransf = FFT2d(img)
    nonZeros = [np.count_nonzero(originalTransf)]

    plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    c = [0, 10, 35, 50, 75, 95]  # 6 different compression levels

    for i in range(2, 7):
        if i == 2:
            compression = c[1]
        elif i == 3:
            compression = c[2]
        elif i == 4:
            compression = c[3]
        elif i == 5:
            compression = c[4]
        elif i == 6:
            compression = c[5]
        nextTransf = originalTransf.copy()
        percentile = np.percentile(
            abs(nextTransf), compression
        )  # threshold the coefficients' magnitudes
        nextTransf = np.where(
            abs(nextTransf) < percentile, 0, nextTransf
        )  # keep only the largest percentile
        nonZeros.append(np.count_nonzero(nextTransf))  # count the number of non zeros
        nextInverse = inverseFFT2d(nextTransf)  # take inverse FT to obtain image

        # Plot compressed image
        plt.subplot(2, 3, i)
        plt.title("{}% compression".format(compression))
        plt.imshow(np.real(nextInverse), cmap="gray")

    print("Number of non zeros:")
    print("Original \t- {} ".format(nonZeros[0]))
    for i in range(1, 6):
        print("{}% Compression - {}".format(c[i], nonZeros[i]))
    plt.show()

def plottingMode():

    size_list = []
    naive_list = []
    fft_list = []
    for size_index in range(6,15,2):
        prob_size = 2**size_index
        size_list.append(prob_size)
        dimension = int(math.sqrt(prob_size))
        rand_values = np.random.random((dimension, dimension))
        naive_runs = []
        fft_runs = []

        for _ in range(10):

            naive_start = time.time()
            naive2d_DFT(rand_values)
            naive_end = time.time()
            naive_time = naive_end - naive_start
            naive_runs.append(naive_time)

            fft_start = time.time()
            FFT2d(rand_values)
            fft_end = time.time()
            fft_time = fft_end - fft_start
            fft_runs.append(fft_time)

        naive_mean = np.mean(np.asarray(naive_runs))
        fft_mean = np.mean(np.asarray(fft_runs))
        naive_var = np.var(np.asarray(naive_runs))
        fft_var = np.var(np.asarray(fft_runs))

        print("For problem size 2^" + str(size_index))
        print("Naive had a mean of: " + str(naive_mean) + " and variance of: " + str(naive_var))
        print("FFT had a mean of: " + str(fft_mean) + " and variance of: " + str(fft_var))
        print("----------")
        naive_list.append(naive_mean)
        fft_list.append(fft_mean)
    
    plt.title("Problem Size vs. Runtime")
    plt.plot(size_list, naive_list, label="Naive", marker="o")
    plt.plot(size_list, fft_list, label="FFT", marker="o")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Default values
    mode = 1
    iFile = "moonlanding.png"

    expSyntax = "fft.py [-m mode] [-i image]"
    syntaxError = 0

    # Parse input with expected format: python fft.py [-m mode] [-i image]
    if len(sys.argv) > 5:
        print("ERROR\tToo many args - Expected Syntax: {}".format(expSyntax))
        exit(1)
    elif len(sys.argv) == 3:
        if sys.argv[1] == "-m":
            mode = sys.argv[2]
        elif sys.argv[1] == "-i":
            iFile = sys.argv[2]
        else:
            syntaxError = 1
    elif len(sys.argv) == 5:
        if sys.argv[1] == "-m" and sys.argv[3] == "-i":
            mode = sys.argv[2]
            iFile = sys.argv[4]
        else:
            syntaxError = 1
    else:
        syntaxError = 1

    if syntaxError:
        print("ERROR\tIncorrect Syntax - Expected Syntax: {}".format(expSyntax))
        exit(1)

    img = cv2.imread(iFile, cv2.IMREAD_GRAYSCALE)
    width = len(img[0])
    height = len(img)

    if int(np.log2(width)) != np.log2(width):
        width = pow(2, int(np.log2(width)) + 1)

    if int(np.log2(height)) != np.log2(height):
        height = pow(2, int(np.log2(height)) + 1)

    img = cv2.resize(img, (width, height))

    if int(mode) == 1:
        fastMode(img)
    elif int(mode) == 2:
        denoiseMode(img)
    elif int(mode) == 3:
        compressionMode(img)
    elif int(mode) == 4:
        plottingMode()
