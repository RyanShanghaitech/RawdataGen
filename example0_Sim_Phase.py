from numpy import *
from matplotlib.pyplot import *
from skimage import data, transform
import mrtrajgen
import nudft
import sdcvd

debug = 1
numPix = 128
ovsImg = 1
objNudft = nudft.NudftClient()

# get img data
img = transform.resize(data.shepp_logan_phantom(), [numPix*ovsImg, numPix*ovsImg]).astype(complex128)

# apply phase map
lstX, lstY = meshgrid(
    linspace(-numPix//2, numPix//2, numPix*ovsImg),
    linspace(-numPix//2, numPix//2, numPix*ovsImg))
getLstX = lambda tht:lstX + tht*(random.randn(*img.shape) - 0.5)
getLstY = lambda tht:lstY + tht*(random.randn(*img.shape) - 0.5)

img *= exp(1j*sqrt((getLstX(1)-2)**2 + (getLstY(1)+5)**2)/50)
img *= exp(1j*(0.6*getLstX(4) - 0.8*getLstY(4) - 6)/100)

# add noise
img += (random.random(img.shape) - 0.5)*1e-3
img += (random.random(img.shape) - 0.5)*1e-3*1j

trjCart_Img = mrtrajgen.genCart(numPix*ovsImg, numPix//2)
trjCart_Ksp = mrtrajgen.genCart(numPix, 0.5)
trjRadial = mrtrajgen.genRadial(linspace(0, 2*pi, ceil(numPix*pi).astype(int64), endpoint=False), linspace(0, 0.5, int64(1.5*numPix/2)))
sr = mrtrajgen.getSlewRate_Circle(1/numPix, 10e-6, 0.5)
trjSpiral, _ = mrtrajgen.genSpiral_Slewrate(
    lambda t: 48*t*0.5/(numPix*pi),
    lambda t: 48*0.5/(numPix*pi),
    lambda t: 0,
    sr,
    10e-6,
    0.5,
    4)
trjSpiral = mrtrajgen.copyTraj(trjSpiral, 48)

lstDsRadial, _ = sdcvd.getDs(trjRadial)
lstDsRadial = sdcvd.fixDs(lstDsRadial, lstDsRadial.shape[1] - 2)
lstDsSpiral, _ = sdcvd.getDs(trjSpiral)
lstDsSpiral = sdcvd.fixDs(lstDsSpiral, int(lstDsSpiral.shape[1]*0.9))

rawdataCart = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjCart_Ksp.reshape(-1, 2)).reshape(trjCart_Ksp.shape[:-1])/(ovsImg**2)
rawdataRadial = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjRadial.reshape(-1, 2)).reshape(trjRadial.shape[:-1])/(ovsImg**2)
rawdataSpiral = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjSpiral.reshape(-1, 2)).reshape(trjSpiral.shape[:-1])/(ovsImg**2)

savez("./Resource/data.npz", img=img, trjCart_Ksp=trjCart_Ksp, trjRadial=trjRadial, trjSpiral=trjSpiral, lstDsRadial=lstDsRadial, lstDsSpiral=lstDsSpiral, rawdataCart=rawdataCart, rawdataRadial=rawdataRadial, rawdataSpiral=rawdataSpiral)

# print shape and dtype
variables = {
    "img": img,
    "trjCart_Ksp": trjCart_Ksp,
    "trjRadial": trjRadial,
    "trjSpiral": trjSpiral,
    "lstDsRadial": lstDsRadial,
    "lstDsSpiral": lstDsSpiral,
    "rawdataCart": rawdataCart,
    "rawdataRadial": rawdataRadial,
    "rawdataSpiral": rawdataSpiral
}
for n, v in variables.items():
    print(f"{n}: shape={v.shape}, dtype={v.dtype}")

# plot
figure()
subplot(121)
imshow(abs(img), cmap="gray")
title("img.abs"); colorbar()
subplot(122)
imshow(angle(img), cmap="hsv", vmin=-pi, vmax=pi)
title("img.ang"); colorbar()

figure()
plot(trjCart_Img.reshape(-1, 2)[:,0], trjCart_Img.reshape(-1, 2)[:,1], marker=".")
title("trjCart_Img"); axis("equal")

figure()
plot(trjRadial.reshape(-1, 2)[:,0], trjRadial.reshape(-1, 2)[:,1], marker=".")
title("trjRadial"); axis("equal")

figure()
plot(trjSpiral.reshape(-1, 2)[:,0], trjSpiral.reshape(-1, 2)[:,1], marker=".")
title("trjSpiral"); axis("equal")

figure()
plot(lstDsRadial.flatten(), marker=".")
title("lstDsRadial")

figure()
plot(lstDsSpiral.flatten(), marker=".")
title("lstDsSpiral")

figure()
plot(abs(rawdataCart).flatten(), marker=".")
title("rawdataCart.abs")

figure()
plot(abs(rawdataRadial).flatten(), marker=".")
title("rawdataRadial.abs")

figure()
plot(abs(rawdataSpiral).flatten(), marker=".")
title("rawdataSpiral.abs")

if debug: show()