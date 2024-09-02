from numpy import *
from numpy.fft import *

def cft(im:ndarray) -> ndarray:
    return fftshift(fftn(fftshift(im)))/asarray(im.shape).prod()
def icft(ks:ndarray) -> ndarray:
    return fftshift(ifftn(fftshift(ks)))*asarray(ks.shape).prod()