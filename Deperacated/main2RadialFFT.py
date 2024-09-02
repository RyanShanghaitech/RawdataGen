from numpy import *
from matplotlib.pyplot import *
import nudft

numPix = 128
objNudft = nudft.NudftClient()

data = load("./Resource/data.npz")
img = data["img"]
trjCart = data["trjCart"]
trjRadial = data["trjRadial"]
lstDsRadial = data["lstDsRadial"]
rawdataRadial = data["rawdataRadial"]

rawdataRadial = rawdataRadial*lstDsRadial

imgRadial0 = objNudft.nuidft(
    rawdataRadial[:,0::2].reshape(-1), 
    trjRadial[:,0::2].reshape(-1, 2), 
    numPix*trjCart.reshape(-1, 2)
    ).reshape(numPix, numPix)
imgRadial1 = objNudft.nuidft(
    rawdataRadial[:,1::2].reshape(-1), 
    trjRadial[:,1::2].reshape(-1, 2), 
    numPix*trjCart.reshape(-1, 2)
    ).reshape(numPix, numPix)

imgReco = imgRadial0 + imgRadial1

figure()
subplot(121)
imshow(real(imgRadial0))
subplot(122)
imshow(imag(imgRadial0))

figure()
subplot(121)
imshow(real(imgRadial1))
subplot(122)
imshow(imag(imgRadial1))

figure()
subplot(121)
imshow(real(imgReco))
subplot(122)
imshow(imag(imgReco))

show()