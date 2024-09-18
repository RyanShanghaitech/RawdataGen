# Try to implement sinc convolution by img-space multiplication
from numpy import *
from matplotlib.pyplot import *
import cft

numPix = 128
idxOri = numPix//2

def getKasserBessel(u):
    L = 6
    B = pi*L/2
    rt = i0(B*sqrt(1-(2*u/L)**2))/L
    if isscalar(rt):
        if isnan(rt): rt = 0
    else:
        rt[isnan(rt)] = 0
    return rt

ker = zeros([numPix, numPix], dtype=complex128)
for idxY in range(numPix):
    for idxX in range(numPix):
        if abs(idxY-idxOri) >= 1 and abs(idxX-idxOri) >= 1 : continue
        coordY = (idxY - idxOri)/numPix
        coordX = (idxX - idxOri)/numPix
        divY = abs(idxY - idxOri)*pi
        divX = abs(idxX - idxOri)*pi
        if divY == 0: divY = 1
        if divX == 0: divX = 1
        ker[idxY,idxX] = exp(1j*(coordX + coordY)*pi)*pi*pi/divY/divX
        if(idxY - idxOri + idxX - idxOri)%2 == 1: ker[idxY,idxX] *= -1
        # if(idxY + idxX)%2 == 0 and (idxY - idxOri + idxX - idxOri) != 0: ker[idxY,idxX] *= -1

msk = cft.ift(ker)

figure(1)
subplot(121)
imshow(abs(ker), vmin=0, vmax=1)
subplot(122)
imshow(angle(ker), vmin=-pi, vmax=pi, cmap="hsv")

figure(2)
subplot(121)
imshow(abs(msk)); colorbar()
subplot(122)
imshow(angle(msk), vmin=-pi, vmax=pi, cmap="hsv"); colorbar()

show()