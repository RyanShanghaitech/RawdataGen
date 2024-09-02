from numpy import *
from matplotlib.pyplot import *
import nudft

numPix = 128
objNudft = nudft.NudftClient()

data = load("./Resource/data.npz")
img = data["img"]
trjCart = data["trjCart"]
rawdataCart = data["rawdataCart"]

imgCart0 = objNudft.nuidft(
    rawdataCart[0::2,:].reshape(-1), 
    trjCart[0::2,:].reshape(-1, 2), 
    numPix*trjCart[numPix*1//4:numPix*3//4,:].reshape(-1, 2)
    ).reshape(numPix//2, numPix)
imgCart1 = objNudft.nuidft(
    rawdataCart[1::2,:].reshape(-1), 
    trjCart[0::2,:].reshape(-1, 2), 
    numPix*trjCart[numPix*1//4:numPix*3//4,:].reshape(-1, 2)
    ).reshape(numPix//2, numPix)

mapPh = zeros([numPix, numPix], dtype=complex128)
for idxY in range(numPix):
    mapPh[idxY,:] = exp(1j*2*pi*(idxY-numPix/2)/numPix)*ones(numPix, dtype=complex128)

imgReco = zeros_like(img)
imgReco[numPix*1//4:numPix*3//4,:] = imgCart0 + mapPh[numPix*1//4:numPix*3//4,:]*imgCart1
imgReco[:numPix*1//4,:] = imgCart0[numPix//4:,:] + mapPh[:numPix*1//4,:]*imgCart1[numPix//4:,:]
imgReco[numPix*3//4:,:] = imgCart0[:numPix//4,:] + mapPh[numPix*3//4:,:]*imgCart1[:numPix//4,:]
# imgReco = imgCart0 + imgCart1

figure()
subplot(121)
imshow(real(imgCart0))
subplot(122)
imshow(imag(imgCart0))

figure()
subplot(121)
imshow(real(imgCart1))
subplot(122)
imshow(imag(imgCart1))

figure()
imshow(angle(mapPh), vmin=-pi, vmax=pi, cmap="hsv")

figure()
subplot(121)
imshow(real(imgReco))
subplot(122)
imshow(imag(imgReco))

show()