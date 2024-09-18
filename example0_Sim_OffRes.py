from numpy import *
from matplotlib.pyplot import *
from skimage import data, transform
import mrtrajgen
import nudft
import sdcvd

constDt = 1 # 0: const Tacq, 1: const dt
debug = 1
numPix = 128
ovsImg = 1
objNudft = nudft.NudftClient()

img = transform.resize(data.shepp_logan_phantom(), [numPix*ovsImg, numPix*ovsImg]).astype(complex128)

# generate trajectory
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

# generate ds
lstDsRadial, _ = sdcvd.getDs(trjRadial)
lstDsRadial = sdcvd.fixDs(lstDsRadial, lstDsRadial.shape[1] - 2)
lstDsSpiral, _ = sdcvd.getDs(trjSpiral)
lstDsSpiral = sdcvd.fixDs(lstDsSpiral, int(lstDsSpiral.shape[1]*0.9))

# generate rawdata
durAcq = 5e-3
maxAngSpeed = 10e-6*9.4*42.58e6*2*pi # 10ppm in 9.4T
mapAngSpeed = maxAngSpeed*(img-0.5) # rad/s

print("generate Cart Rawdata")
dt = 10e-6 if constDt else durAcq/trjCart_Ksp.shape[1]
rawdataCart = zeros(trjCart_Ksp.shape[:-1], dtype=complex128)
for idxRO in range(trjCart_Ksp.shape[1]):
    rawdataCart[:,idxRO] = objNudft.nudft(
        (img*exp(1j*mapAngSpeed*(idxRO*dt + 1e-3))).reshape(-1), 
        trjCart_Img.reshape(-1, 2), 
        trjCart_Ksp[:,idxRO,:].reshape(-1, 2)
        )/(ovsImg**2)

print("generate Radial Rawdata")
dt = 10e-6 if constDt else durAcq/trjRadial.shape[1]
rawdataRadial = zeros(trjRadial.shape[:-1], dtype=complex128)
for idxRO in range(trjRadial.shape[1]):
    rawdataRadial[:,idxRO] = objNudft.nudft(
        (img*exp(1j*mapAngSpeed*(idxRO*dt + 1e-3))).reshape(-1), 
        trjCart_Img.reshape(-1, 2), 
        trjRadial[:,idxRO,:].reshape(-1, 2)
        )/(ovsImg**2)

print("generate Spiral Rawdata")
dt = 10e-6 if constDt else durAcq/trjSpiral.shape[1]
rawdataSpiral = zeros(trjSpiral.shape[:-1], dtype=complex128)
for idxRO in range(trjSpiral.shape[1]):
    rawdataSpiral[:,idxRO] = objNudft.nudft(
        (img*exp(1j*mapAngSpeed*(idxRO*dt + 1e-3))).reshape(-1), 
        trjCart_Img.reshape(-1, 2), 
        trjSpiral[:,idxRO,:].reshape(-1, 2)
        )/(ovsImg**2)
    
# save rawdata
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
subplot(121)
imshow(abs(img), cmap="gray")
colorbar()
subplot(122)
imshow(real(mapAngSpeed))
colorbar()

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