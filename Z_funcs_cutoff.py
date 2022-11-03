#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:27:52 2022

@author: s2078322
"""


import numpy as np
from scipy import special

import matplotlib.pyplot as plt


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
        
#%% K matrices (=M matrices at leading order)

def getLambda(a0):
    return 16 * TWOPI * a0

def getK0(E,a,a0):
    lam=getLambda(a0)
    k0=lam*(-1+a**2*E**2/12)/(4*TWOPI2) # extra factor of 1/4pi from decomposing a second time
    return k0



#%% Z functions



def summandZ00(qSQ,  nvecs):
    """Summand."""
    rSQs = (nvecs**2).sum(1)
    Ds = qSQ-rSQs
    return np.exp(alphaKSS*Ds)/Ds

def T1(Qsq):
    """T1."""
    rng = range(-C1cut, C1cut+1)
    mesh = np.meshgrid(*([rng] * 3))
    nvecs = np.vstack([y.flat for y in mesh]).T 
    return np.sum(summandZ00(Qsq, nvecs))

def T2(Qsq):
    """T2."""
    x=np.sqrt(np.abs(Qsq*alphaKSS))
    if Qsq >= 0:
        ttmp = 2 *np.exp(x**2)*np.pi**(3/2) *(-1+2*x*special.dawsn(x))/(np.sqrt(alphaKSS))
    else:
        ttmp = TWOPI*(np.exp(-x**2)*np.sqrt(np.pi)+np.pi*x*special.erf(x))/(np.sqrt(alphaKSS))
    return ttmp


        


#%% F functions

def sum_int(qsq,L):
    """Regulated F00 function."""
    Qsq=(L/TWOPI)**2*qsq
    summation=T1(Qsq)
    integral=T2(Qsq)
    return 1/(TWOPI*L)*(summation-integral)


def getF00(E,L,a):
    qsq=E**2/4-1 +a**2/120 *(E**4-3*E**2+6)
    return 1/2 * 1/(4*TWOPI2) *(1+a**2/6* (2/5 * E**2-3/5))/(2*E)*sum_int(qsq,L)
#%%quantisation condition
def getM(E,L,a,a0):
       
    k0=getK0(E,a,a0)
    f00=getF00(E, L,a)
    return k0-1/f00

def getMp(Estar,L,theta,l,a=float("NaN"),nPvec=np.array([0.0,0.0,0.0]),a4=float("NaN")):
    return(getM(Estar+delta,L,theta,l,a,nPvec,a4)-getM(Estar-delta,L,theta,l,a,nPvec,a4))/(2*delta)

#%%Getting energy levels
L=5 #Mpi


for a0 in [-10,-1,-0.1,0,0.1,1,10]:
    lam=getLambda(a0)
    if (True):
        
        E0=[]
        E1=[]
        E2=[]
        E3=[]
        
        E0a=[]
        E1a=[]
        E2a=[]
        E3a=[]
        Evals=np.arange(2.1,12.5,0.005)
        a_s=np.arange(0,1,0.1)
        for a in a_s:
            #print(a)
            
            M0a=[]
            M0=[]
            for Estar in Evals:
                        
                        M0a=M0a+[getM(Estar,L,a,a0)]
                        M0=M0+[getM(Estar,L,0,a0)]
        #    plt.plot(Evals,M0) 
        #    plt.show()
            rootsa=getzeros(M0a,Evals,700)
            #print(rootsa)
            roots=getzeros(M0,Evals,300)
            
            E0=E0+[roots[0]]
            E1=E1+[roots[1]]
            E2=E2+[roots[2]]
            E3=E3+[roots[3]] 
            
            
            E0a=E0a+[rootsa[0]]
            E1a=E1a+[rootsa[1]]
            E2a=E2a+[rootsa[2]]
            E3a=E3a+[rootsa[3]] 
        plt.figure(figsize=(8,4))    
        plt.plot(a_s,E0,'C0--',a_s,E0a,'C0')
        plt.plot(a_s,E1,'C1--',a_s,E1a,'C1')
        plt.plot(a_s,E2,'C2--',a_s,E2a,'C2')
        plt.plot(a_s,E3,'C3--',a_s,E3a,'C3')
        plt.ylabel("Energy/ $m_\pi$")
        plt.xlabel("$m_\pi$ a")
        plt.title("$m_\pi$L=" + str(L) + " $m_\pi$a0="+ str(a0)+ " $\lambda$="+ str(round(lam,0)))
        plt.savefig("L=" + str(L) + " a0="+ str(a0)+".png")
        plt.savefig("L=" + str(L) + " a0="+ str(a0)+".pdf")
        plt.show()

if(False):

    L=5 #Mpi
    a0s=[-1000,-1,-0.5,-0.1,0,0.1,0.5,1]
    for a0 in a0s:
        print(a0)
        Evals=np.arange(2.1,12.5,0.005)
        M0=[]
        for Estar in Evals:
            M0=M0+[getM(Estar,L,0,a0)]
        print(getzeros(M0,Evals,300))
        
