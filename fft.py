import sys
import numpy as np

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
    arr = np.asarray(arr)
    transform = [0] * len(arr)
    n_values = [0] * len(arr)
    for i in range(len(arr)):
        n_values[i] = i

    n_values = np.asarray(n_values)
    for i in range(len(arr)):
        transform[i] = np.sum(arr*np.exp((-2j*np.pi)/len(arr)*i*n_values))

    print(transform)
    return np.asarray(transform)

if __name__ == "__main__":
    print(naiveDFT([2,3,4,5]))