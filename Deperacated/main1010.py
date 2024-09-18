# explore the equivalent convolution kernel when dividing the F-1(KB kernel) in image domain.
from numpy import *
from matplotlib.pyplot import *
import cft

sizImg = 128
sizWind = 6

def getKasserBessel(u):
    L = 6
    B = pi*L/2
    rt = i0(B*sqrt(1-(2*u/L)**2))/L
    if isscalar(rt):
        if isnan(rt): rt = 0
    else:
        rt[isnan(rt)] = 0
    return rt

kspKernel = zeros((sizImg, sizImg), dtype=float64)
ceilSizWind = int64(ceil(sizWind))
for idxDkx in range(-ceilSizWind//2, ceilSizWind//2):
    for idxDky in range(-ceilSizWind//2, ceilSizWind//2):
        idxKx = idxDkx+sizImg//2
        idxKy = idxDky+sizImg//2
        kspKernel[idxKy, idxKx] = getKasserBessel(idxDkx)*getKasserBessel(idxDky)
imgMask = cft.ift(kspKernel)

imgMask_Inv = 1/imgMask
kspKernel_Inv = cft.cft(imgMask_Inv)

figure()
subplot(121)
imshow(abs(kspKernel))
subplot(122)
imshow(abs(imgMask))

figure()
subplot(121)
imshow(abs(kspKernel_Inv))
subplot(122)
imshow(abs(imgMask_Inv))

show()