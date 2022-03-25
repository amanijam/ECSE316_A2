import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


smallSize = 16

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
    if sys.argv[1] == "-m":
        mode = sys.argv[2]
    elif sys.argv[1] == "-i":
        iFile = sys.argv[2]
    else:
        syntaxError = 1
elif len(sys.argv) == 5:
    print("4 args (mode and image)")
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


if __name__ == "__main__":

    # x = [[2, 3, 4], [4, 3, 5], [7, 5, 6]]

    # print(naive2d_DFT(x))
    # print(FFT2d(x))
    # print(np.fft.fft2(x))

    img = cv2.imread(iFile, cv2.IMREAD_GRAYSCALE)
    width = len(img[0])
    height = len(img)

    if int(np.log2(width)) != np.log2(width):
        width = pow(2, int(np.log2(width)) + 1)

    if int(np.log2(height)) != np.log2(height):
        height = pow(2, int(np.log2(height)) + 1)

    img = cv2.resize(img, (width, height))

    if mode == 1:
        fastMode(img)
