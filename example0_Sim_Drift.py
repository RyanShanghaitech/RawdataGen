from numpy import *
from matplotlib.pyplot import *
from skimage import data, transform
import mrtrajgen
import nudft
import sdcvd
import cft

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
trjSpiral, lstGrad = mrtrajgen.genSpiral_Slewrate(
    lambda t: 48*t*0.5/(numPix*pi),
    lambda t: 48*0.5/(numPix*pi),
    lambda t: 0,
    sr,
    10e-6,
    0.5,
    4)
trjSpiral = mrtrajgen.copyTraj(trjSpiral, 48)

# allocate
lstGrad = concatenate([zeros_like(lstGrad), lstGrad], axis=0, dtype=complex128)
ftLstGrad = zeros_like(lstGrad, dtype=complex128)

# transform to frequency domain
ftLstGrad[:,0] = cft.fft(lstGrad[:,0])
ftLstGrad[:,1] = cft.fft(lstGrad[:,1])

# apply transfer function
tau = trjSpiral.shape[0]//10 # unit: dt
ftLstGrad[:,0] = ftLstGrad[:,0]/(1+1j*linspace(-pi, pi, ftLstGrad.shape[0])*tau)
ftLstGrad[:,1] = ftLstGrad[:,1]/(1+1j*linspace(-pi, pi, ftLstGrad.shape[0])*tau)

# transform to time domain
lstGrad_Delay = zeros_like(lstGrad, dtype=complex128)
lstGrad_Delay[:,0] = cft.ift(ftLstGrad[:,0])
lstGrad_Delay[:,1] = cft.ift(ftLstGrad[:,1])

# remove t<0 part
lstGrad = lstGrad[-lstGrad.shape[0]//2:,:].real
lstGrad_Delay = lstGrad_Delay[-lstGrad_Delay.shape[0]//2:,:].real

# regenerate trajectory
trjSpiral_Delay = mrtrajgen.tranGrad2Traj_MinSR(lstGrad_Delay, 10e-6)
trjSpiral_Delay = mrtrajgen.copyTraj(trjSpiral_Delay, 48)

# sample density
lstDsRadial, _ = sdcvd.getDs(trjRadial)
lstDsRadial = sdcvd.fixDs(lstDsRadial, lstDsRadial.shape[1] - 2)
lstDsSpiral, _ = sdcvd.getDs(trjSpiral)
lstDsSpiral = sdcvd.fixDs(lstDsSpiral, int(lstDsSpiral.shape[1]*0.9))

# generate rawdata
rawdataCart = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjCart_Ksp.reshape(-1, 2)).reshape(trjCart_Ksp.shape[:-1])/(ovsImg**2)
rawdataRadial = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjRadial.reshape(-1, 2)).reshape(trjRadial.shape[:-1])/(ovsImg**2)
rawdataSpiral = objNudft.nudft(img.reshape(-1), trjCart_Img.reshape(-1, 2), trjSpiral_Delay.reshape(-1, 2)).reshape(trjSpiral_Delay.shape[:-1])/(ovsImg**2)

# save
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
subplot(211)
plot(lstGrad, marker=".")
title("lstGrad")
subplot(212)
plot(lstGrad_Delay, marker=".")
title("lstGrad_Delay")

figure()
plot(trjSpiral[0,:,0], trjSpiral[0,:,1], marker=".", label="Spiral")
plot(trjSpiral_Delay[0,:,0], trjSpiral_Delay[0,:,1], marker=".", label="Spiral_Delay")
title("trjSpiral"); axis("equal"); legend()

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