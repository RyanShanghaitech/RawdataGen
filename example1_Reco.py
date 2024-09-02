from numpy import *
from matplotlib.pyplot import *
import nudft
import mrtrajgen

numPix = 128
data = load("./Resource/data.npz")
objNudft = nudft.NudftClient()

img=data["img"]
trjCart_Ksp=data["trjCart_Ksp"]
trjRadial=data["trjRadial"]
trjSpiral=data["trjSpiral"]
lstDsRadial=data["lstDsRadial"]
lstDsSpiral=data["lstDsSpiral"]
rawdataCart=data["rawdataCart"]
rawdataRadial=data["rawdataRadial"]
rawdataSpiral=data["rawdataSpiral"]

trjCart_Img = mrtrajgen.genCart(numPix, numPix//2)
imgReco_Cart = objNudft.nuidft(rawdataCart.reshape(-1), trjCart_Ksp.reshape(-1, 2), trjCart_Img.reshape(-1, 2)).reshape(numPix, numPix)
imgReco_Radial = objNudft.nuidft(rawdataRadial.reshape(-1)*lstDsRadial.reshape(-1), trjRadial.reshape(-1, 2), trjCart_Img.reshape(-1, 2)).reshape(numPix, numPix)
imgReco_Spiral = objNudft.nuidft(rawdataSpiral.reshape(-1)*lstDsSpiral.reshape(-1), trjSpiral.reshape(-1, 2), trjCart_Img.reshape(-1, 2)).reshape(numPix, numPix)

figure()
imshow(abs(imgReco_Cart), cmap="gray")
title("imgReco_Cart.abs")

figure()
imshow(abs(imgReco_Radial), cmap="gray")
title("imgReco_Radial.abs")

figure()
imshow(abs(imgReco_Spiral), cmap="gray")
title("imgReco_Spiral.abs")

show()