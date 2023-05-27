import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

hermitian = lambda npArr : npArr.conj().transpose(); 

def draw( x, y, xAxisTitle, yAxisTitle, title ):
    fig, axs = plt.subplots()
    axs.set_title(title)
    axs.plot(x, y)
    axs.set_xlabel(xAxisTitle)
    axs.set_ylabel(yAxisTitle)

def draw_3d(x,y,z, xAxisTitle, yAxisTitle, zAxisTitle):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label="signal 1")
    ax.legend()
    ax.set_xlabel(xAxisTitle)
    ax.set_ylabel(yAxisTitle)
    ax.set_zlabel(zAxisTitle)
    plt.show()


def complexAGWN(variance, size):
    n = size[0]
    col = lambda : np.random.normal(loc=0, scale= variance*1/np.sqrt(2), size=(n,2)).view(np.complex)
    return np.hstack([ col()/np.sqrt(2) for j in range(size[1]) ])

def awgn(x,snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x+noise

def toep(u):
    def column(u,k):
        return cp.vstack( u[i-k] if i-k >= 0 else cp.conj(u[k-i]) for i in range(u.shape[0]))
    return cp.hstack([column(u, j) for j in range(u.shape[0])])

def RMSE(estimates, labels):
  if (len(estimates) != len(labels)):
    return -1
  squareError = (np.array(labels) - np.array(estimates))**2
  return np.sqrt(np.mean(squareError))

def findNMaxPeaks(y ,list, n):
    sortedPeaks = np.argsort(y[list])[::-1]
    return sortedPeaks[0:n]

