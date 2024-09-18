from numpy import *
from numpy.fft import *

def fft(im:ndarray) -> ndarray:
    return fftshift(fftn(fftshift(im)))/asarray(im.shape).prod()
def ift(ks:ndarray) -> ndarray:
    return fftshift(ifftn(fftshift(ks)))*asarray(ks.shape).prod()