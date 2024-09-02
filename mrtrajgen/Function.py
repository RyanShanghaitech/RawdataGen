from numpy import *
from typing import Callable
from .Utility import tranGrad2Traj_MinSR

def genSpiral_DeltaK(getDeltaK:Callable, getDrhoDtht:Callable, phase:int|float=0, rhoMax:float|int=0.5) -> ndarray:
    """
    # description:
    generate spiral sampling trajectory, subject to sampling interval

    # parameter:
    `getDeltaK`:Callable: function of sampling interval with respect to rho and theta, e.g. `lambda rho, tht: dNyq` for spiral with constant Nyquist sampling interval, in `/pix`
    `getDrhoDtht`:Callable: function of dRho/dTheta with respect to rho and theta, e.g. `lambda rho, tht: b` for Archimedean spiral, `lambda rho, tht: rho + 1` for variable density spiral, in `/pix/rad`
    `phase`: phase controlling rotation of spiral, in `rad`
    `rhoMax`: maximum value of rho, in `/pix`

    # return:
    kspace trajectory: [[kx1, ky1], [kx2, ky2], ..., [kxn, kyn]], in `/pix`
    """
    lstTht = array([0, 2*pi])
    lstRho = array([0, getDeltaK(0, 0)])
    while True:
        dK = getDeltaK(lstRho[-1], lstTht[-1])
        quoDrhoDtht = getDrhoDtht(lstRho[-1], lstTht[-1])
        
        dTht = dK/sqrt(quoDrhoDtht**2 + lstRho[-1]**2)
        dRho = dTht*quoDrhoDtht
        
        # append new point
        thtNew = lstTht[-1]+dTht
        rhoNew = lstRho[-1]+dRho
        if(rhoNew < rhoMax):
            lstTht = append(lstTht, thtNew)
            lstRho = append(lstRho, rhoNew)
        else:
            break

    lstKx = lstRho*cos(lstTht + phase)
    lstKy = lstRho*sin(lstTht + phase)
    lstKxKy = array([lstKx, lstKy]).T

    return lstKxKy

def genSpiral_Slewrate(getD0RhoTht:Callable, getD1RhoTht:Callable, getD2RhoTht:Callable, sr:int|float, dt:int|float, kmax:int|float, oversamp:int|float=2, flagDebugInfo:bool=False, gamma:int|float=42.58e6) -> tuple[ndarray, ndarray]:
    '''
    # description
    generate spiral trajectory, subject to slew rate

    # parameter
    `getD0RhoTht`: function of 0th order derivation of rho with respect to theta, in `/pix`
    `getD1RhoTht`: function of 1st order derivation of rho with respect to theta, in `/pix/rad`
    `getD2RhoTht`: function of 2nd order derivation of rho with respect to theta, in `/pix/rad^2`

    # return
    kspace trajectory: [[kx1, ky1], [kx2, ky2], ..., [kxn, kyn]], in `/pix`
    gradient list: [[gx1, gy1], [gx2, gy2], ..., [gxn, gyn]], in `T/pix`
    '''
    sovQDF = lambda a, b, c: (-b+sqrt(max(b**2-4*a*c, 0)))/(2*a)
    lstTht = empty([0], dtype=float64)
    lstRho = empty([0], dtype=float64)

    d0ThtTime = 0
    d1ThtTime = 0
    d2ThtTime = 0
    d0RhoTht = getD0RhoTht(d0ThtTime); assert(d0RhoTht == 0)
    d1RhoTht = getD1RhoTht(d0ThtTime)
    d2RhoTht = getD2RhoTht(d0ThtTime)
    while d0RhoTht < kmax and lstRho.size < 1e6:
        a = d0RhoTht**2 + d1RhoTht**2
        b = 2*d0RhoTht*d1RhoTht*d1ThtTime**2 + 2*d1RhoTht*d2RhoTht*d1ThtTime**2
        c = d0RhoTht**2*d1ThtTime**4 - 2*d0RhoTht*d2RhoTht*d1ThtTime**4 + 4*d1RhoTht**2*d1ThtTime**4 + d2RhoTht**2*d1ThtTime**4 - sr**2*gamma**2

        d2ThtTime = sovQDF(a, b, c)
        d1ThtTime += d2ThtTime*(dt/oversamp)
        d0ThtTime += d1ThtTime*(dt/oversamp)
        d0RhoTht = getD0RhoTht(d0ThtTime)
        d1RhoTht = getD1RhoTht(d0ThtTime)
        d2RhoTht = getD2RhoTht(d0ThtTime)

        lstTht = append(lstTht, d0ThtTime)
        lstRho = append(lstRho, d0RhoTht)

        if flagDebugInfo: print(f"rho = {d0RhoTht}/{kmax}")

    lstRho = lstRho[::oversamp]
    lstTht = lstTht[::oversamp]
    lstTraj_Ideal = array([
        lstRho*cos(lstTht),
        lstRho*sin(lstTht)]).T
    lstGrad = array([
        (lstTraj_Ideal[1:,0] - lstTraj_Ideal[:-1,0])/dt/gamma,
        (lstTraj_Ideal[1:,1] - lstTraj_Ideal[:-1,1])/dt/gamma]).T
    lstTraj = tranGrad2Traj_MinSR(lstGrad, dt)

    return lstTraj, lstGrad

def genRadial(lstTht:ndarray, lstRho:ndarray) -> ndarray:
    """
    # description:
    generate radial sampling trajectory

    # parameter:
    `lstTht`: list of theta of spokes, in `rad`
    `lstRho`: list of rho of spokes, in `/pix`

    # return:
    kspace trajectory: [[kx1, ky1], [kx2, ky2], ..., [kxn, kyn]], in `/pix`
    """
    # shape check
    assert(size(lstTht.shape) == 1)
    assert(size(lstRho.shape) == 1)
    
    # generate kspace trajectory
    lstKx = [lstRho*cos(lstTht[idxTht]) for idxTht in range(lstTht.size)]
    lstKy = [lstRho*sin(lstTht[idxTht]) for idxTht in range(lstTht.size)]

    return array([lstKx, lstKy]).transpose([1,2,0])

def genCart(numPt:int|float, max:int|float=0.5) -> ndarray:
    """
    # description:
    generate cartesian sampling trajectory

    # parameter:
    `numPt`: number of point in one dimension
    `max`: maximum coordinate value

    # return:
    trajectory: [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    # shape check
    lstKx, lstKy = meshgrid(
        linspace(-max, max, numPt, endpoint=False),
        linspace(-max, max, numPt, endpoint=False))
    return array([lstKx.flatten(), lstKy.flatten()]).T.reshape(numPt, numPt, 2)
    