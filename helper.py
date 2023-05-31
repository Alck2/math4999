import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Input: numpy array
# Ouput: numpy array
# Returns the conjugate transpose of a numpy array
hermitian = lambda npArr : npArr.conj().transpose(); 

# Input:
#   x: arrayType
#   y: arrayType
#   xAxisTitle: string
#   yAxisTitle: string
#   title: string
# Draw a simple 2d line graph for the given x and y with xAxisTitle, yAxisTitle, and title
def draw( x, y, xAxisTitle, yAxisTitle, title ):
    fig, axs = plt.subplots()
    axs.set_title(title)
    axs.plot(x, y)
    axs.set_xlabel(xAxisTitle)
    axs.set_ylabel(yAxisTitle)

# Input:
#   x: arrayType
#   y: arrayType
#   z: arrayType
#   xAxisTitle: string
#   yAxisTitle: string
#   zAxisTitle: string
#   title: string
# Draw a simple 3d line graph for the given x, y, and z with xAxisTitle, yAxisTitle, zAxisTitle, and title
def draw_3d(x,y,z, xAxisTitle, yAxisTitle, zAxisTitle):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label="signal 1")
    ax.legend()
    ax.set_xlabel(xAxisTitle)
    ax.set_ylabel(yAxisTitle)
    ax.set_zlabel(zAxisTitle)
    plt.show()

# # Input:
# #   x: 1d numpy array
# #   snr: double
# # Output: 1d numpy array with same size as x
# # Given signal x, returns the signal with Additive Gaussian White Noise of Signal-to-Noise Ration snr.
# def awgn(x,snr):
#     snr = 10 ** (snr / 10.0)
#     xpower = np.sum(x**2) / len(x)
#     npower = xpower / snr
#     noise = np.random.randn(len(x)) * np.sqrt(npower)
#     return x+noise

# Input:
#   estimates: arrayType
#   labels: arrayType
# Output: double
# Returns the root-mean-square error between the estimates and labels if estimates and labels are with the same size.
# -1 otherwise.
def RMSE(estimates, labels):
  if (len(estimates) != len(labels)):
    return -1
  squareError = (np.array(labels) - np.array(estimates))**2
  return np.sqrt(np.mean(squareError))

# Input:
#   y: ArrayType
#   list: ArrayType
#   n: integer
# Output: ArrayType
# Return the top n highest peaks according to y
def findNMaxPeaks(y ,list, n):
    sortedPeaks = np.argsort(y[list])[::-1]
    return sortedPeaks[0:n]


from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
def awgn(s,SNRdB,L=1):
  """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
  """
  gamma = 10**(SNRdB/10) #SNR to linear scale
  if s.ndim==1:# if s is single dimensional vector
      P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
  else: # multi-dimensional signals like MFSK
      P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
  N0=P/gamma # Find the noise spectral density
  if isrealobj(s):# check if input is real/complex object type
      n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
  else:
      n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
  r = s + n # received signal
  return r