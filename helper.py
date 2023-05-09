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
    return np.hstack([ col() for j in range(size[1]) ])

def awgn(s,SNRdB):
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1: # if s is single dimensional vector
        try:
            P=np.sum(np.abs(s)**2)/len(s) #Actual power in the vector
        except:
            P=np.sum(np.abs(s)**2)/len(s)
    else: # multi-dimensional signals like MFSK
        P=np.sum(np.sum(np.abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    n = np.sqrt(N0/2)*(np.random.standard_normal(s.shape)+1j*np.random.standard_normal(s.shape))
    return n

def toep(u):
    def column(u,k):
        return cp.vstack( u[i-k] if i-k >= 0 else cp.conj(u[k-i]) for i in range(u.shape[0]))
    return cp.hstack([column(u, j) for j in range(u.shape[0])])

