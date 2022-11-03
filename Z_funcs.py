#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:27:52 2022

@author: s2078322
"""


import numpy as np

from scipy.special import erfi
from scipy.special import erf





TWOPI = 2.0*np.pi
TWOPI2 = 4.0*np.pi**2

#numerical parameters
alphaKSS = 0.1
C1cut = int(4/np.sqrt(alphaKSS))
C1cut = 15

delta=0.000001

mk=495.7/135

def getzeros(y,x,lim):
    y=np.array(y)
    x=np.array(x)
    if x.size != y.size:
        print("ERROR, INCOMPATIBLE")
        return float("nan")
    y[y>lim]=float("NaN")
    y[y<-lim]=float("NaN")
    
    zeros=[]
    for i in range(0,y.size-1):
        if ((y[i]<0) and (y[i+1]>0)) or ((y[i]>0) and (y[i+1]<0)):
            root=x[i]-y[i]* (x[i+1]-x[i])/(y[i+1]-y[i])
            zeros=zeros+[root]
        elif (y[i]==0):
            root=x[i]
            zeros=zeros+[root]
        
    
    zeros=np.array(zeros)
    return zeros
        
#%% K matrices
    
#k0 is used to get the energy levels
#k4 is used to get accidental degeneracy
#k2 is only used for chipt
    
def getK0(Estar,a=float("NaN")):

    if np.isnan(a):
        #del0=-0.199833 - 0.00110613*(Estar - mk)
        del0= 0.660389  + 0.0027793*(Estar -  mk)
        k0=16*np.pi*mk/np.sqrt(mk**2/4-1)*np.tan(del0)
    else:
        k0=8*TWOPI*Estar*(-a) 
    return k0



#%% Z functions

def getZ00prime(qSQ, L, nPvec,theta):
    Z00prime=(getZ00(qSQ+delta, L, nPvec,theta)-getZ00(qSQ, L, nPvec,theta))/delta
    return Z00prime

def summandZ00(qSQ, L, nPvec, nvecs):
    """Summand."""
    pSQ = qSQ*TWOPI2/L**2
    EstarSQ = 4.0*(1.0+pSQ)
    nPSQ = nPvec@nPvec
    nPmag = np.sqrt(nPSQ)
    PSQ = TWOPI2*nPSQ/L**2
    ESQ = EstarSQ+PSQ
    gamSQ = ESQ/EstarSQ
    if nPmag == 0.0:
        rSQs = (nvecs**2).sum(1)
    else:
        npars = ((nvecs*nPvec).sum(1))/nPmag
        rparSQs = (npars-nPmag/2.0)**2/gamSQ
        nkhat = nPvec/nPmag
        nparnkhats = np.dot(np.transpose([npars]), [nkhat])
        rperpSQs = ((nvecs-nparnkhats)**2).sum(1)
        rSQs = rparSQs+rperpSQs
    Ds = rSQs-qSQ
    return np.exp(-alphaKSS*Ds)/Ds

def T1(qSQ, L, nPvec,theta):
    """T1."""
    rng = range(-C1cut, C1cut+1)
    mesh = np.meshgrid(*([rng] * 3))
    nvecs = np.vstack([y.flat for y in mesh]).T +theta
    return np.sum(summandZ00(qSQ, L, nPvec, nvecs))\
        / np.sqrt(2.0*TWOPI)

def T2(qSQ, L, nPvec):
    """T2."""
    pSQ = qSQ*TWOPI2/L**2
    EstarSQ = 4.0*(1.0+pSQ)
    nPSQ = nPvec@nPvec
    PSQ = TWOPI2*nPSQ/L**2
    ESQ = EstarSQ+PSQ
    gamSQ = ESQ/EstarSQ
    gamma = np.sqrt(gamSQ)
    if qSQ >= 0:
        ttmp = 2.0*(np.pi**2)*np.sqrt(qSQ)\
             * erfi(np.sqrt(alphaKSS*qSQ))\
             - 2.0*np.exp(alphaKSS*qSQ)\
             * np.sqrt(np.pi**3)/np.sqrt(alphaKSS)
    else:
        ttmp = -2.0*(np.pi**2)*np.sqrt(-qSQ)\
             * erf(np.sqrt(-alphaKSS*qSQ))\
             - 2.0*np.exp(alphaKSS*qSQ)\
             * np.sqrt(np.pi**3)/np.sqrt(alphaKSS)
    return gamma*ttmp/np.sqrt(2.0*TWOPI)

def getZ00(qSQ, L, nPvec,theta):
    """Z function."""
    return T1(qSQ, L, nPvec,theta)\
        + T2(qSQ, L, nPvec)
        


#%% F functions

def getF00(E, L, nPvec,theta):
    """Regulated F00 function."""
    nPSQ = nPvec@nPvec
    PSQ = TWOPI2*nPSQ/L**2
    ESQ = E**2
    EstarSQ = ESQ-PSQ
    Estar = np.sqrt(EstarSQ)
    qSQ = L**2*(EstarSQ/4.0-1.0)/TWOPI2
    gamma = np.sqrt(ESQ/EstarSQ)
    Z00 = getZ00(qSQ, L, nPvec,theta)
    return -Z00/(8.0*Estar*L*np.pi**(3.0/2.0)*gamma)

def getF00prime( E, L, nPvec,theta):
    """Regulated F40 function."""
    ll = getF00(E+delta, L, nPvec,theta)
    rr = getF00(E, L, nPvec,theta)
    return (ll-rr)/delta

#%%quantisation condition
def getM(E,L,theta,l,a=float("NaN"),nPvec=np.array([0.0,0.0,0.0]),a4=float("NaN")):
    Estar=np.sqrt(E**2-np.sum(nPvec**2)*TWOPI2/L**2)
    
    k0=getK0(Estar,a)
    f00=getF00(E, L,nPvec ,theta)
    

    if l==4:
        k4=getK4(Estar,a4)
        f44=getF44(Estar,L,theta)
        f40=getF40(Estar, L,theta)
        denom=f00*f44-f40**2
        return (1+f44*k4-f40**2 *k0*k4 +f00*(k0+f44*k0*k4))/denom

    elif l==0:
        return k0+1/f00

def getMp(Estar,L,theta,l,a=float("NaN"),nPvec=np.array([0.0,0.0,0.0]),a4=float("NaN")):
    return(getM(Estar+delta,L,theta,l,a,nPvec,a4)-getM(Estar-delta,L,theta,l,a,nPvec,a4))/(2*delta)




