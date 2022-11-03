#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we will be doing phi4 theory with s-wave

we set m=1
"""

import numpy as np
from matplotlib import rc

import matplotlib.pyplot as plt
from scipy.special import erfi
from scipy import optimize
from scipy.special import dawsn
import cmath


TWOPI=2*np.pi
PI=np.pi

#Dynamical quantities



#Regulaarisation constants
Damping=0.5
cut=4
lcut=int(cut/np.sqrt(Damping))


def Etop(E,a):
    q2=E**2/4-1+2*a**2
    if q2>=0:
        return np.sqrt(q2)
    elif q2<0:
        return np.sqrt(-q2)*(1j)
def ptoE(q,a):
    return 2*np.sqrt(q**2+1-2*a**2)

def scatlentolam(scat):
    return -32*np.pi*scat


def getzeros(y,x,lim):
    y=np.real(np.array(y))
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
        


def getsum(p,a,mL):
        
    rng = range(-lcut, lcut+1)
    mesh = np.meshgrid(*([rng] * 3))
    nvecs = np.vstack([y.flat for y in mesh]).T    
    nvecs=nvecs *2*PI/mL
    
    nSQs=(nvecs**2).sum(1)
    sqdiff=p**2-nSQs
    
    alpha=Damping*(TWOPI/mL)**2
    
    denom=sqdiff +a**2 *2/5 *(4*p**4+5*p**2)  
    summand=1/mL**3 *1/denom * np.exp(alpha*sqdiff)

    where_are_NaNs = np.isnan(summand)
    summand[where_are_NaNs] = 0
    sum_total=sum(summand)
    
    return sum_total

def getintegral(p,a,mL):
    
    alpha=Damping*(TWOPI/mL)**2
    
    oldint=- 2*np.exp(p**2*alpha)*PI**(3/2)/np.sqrt(alpha)+2*p*PI**2*erfi(p*np.sqrt(alpha))
    newint=4/5 *np.exp(p**2*alpha)*p*(5+4*p**2)*PI**(3/2)*(p*np.sqrt(alpha)+(1-2*p**2*alpha)*dawsn(p*np.sqrt(alpha)))
    return 1/(TWOPI**3)*(oldint-a**2 *newint)

def getG00(E,a,mL):   
    p2=Etop(E,a)**2
    return 1/(4*np.sqrt(1+p2)) *(1+a**2 *(15+26*p2+16*p2*p2)/(5+p2))
    
def getF00(E,a,mL):
    
    p=Etop(E,a)
    
    prefactor =1/2 * 1/(4*PI)*getG00(E,a,mL)
    
    summand=getsum(p,a,mL)
    integral=getintegral(p,a,mL)
    
    
    return prefactor*(summand-integral)

def getK(E,a,lam,mL):
    p=Etop(E,a)
    return lam*(1-a**2*(31/3+4*p**2))

def getQC(E,a,lam,mL):
    
    #return getK(E,a,lam,mL)+1/getF00(E,a,mL)
    return 1/getF00(E,a,mL)+getK(E,a,lam,mL)

def afit(a,x,y):
    return x* a**2+y


#%% plot teh qc as a check
if(False):
    print(cut)
    lam=-100
    mL=5
    a=0.3

    Evals=np.arange(1.5,5,0.0023)
    
    f00=[]
    f00a=[]
    
    qc=[]
    qca=[]
    
    for e in Evals:
        #print(e,end=' ')
        f00=f00+[getF00(e,0,mL)]
        f00a=f00a+[getF00(e,a,mL)]
        qc=qc+[getQC(e,0,lam,mL)]
        qca=qca+[getQC(e,a,lam,mL)]
        
    f00=np.array(f00)
    f00a=np.array(f00a)
    
        
    
    
    lim=0.3
    f00[np.abs(f00)>3*lim]=float('NaN')
    f00a[np.abs(f00a)>2*lim]=float('NaN')
    
#    plt.plot(qtok(Etoq(Evals,0),mL)**2,f00)
#    plt.plot(qtok(Etoq(Evals,0),mL)**2,f00a)

    plt.plot(Evals,f00)
    plt.plot(Evals,f00a)
    
    plt.plot([Evals[0],Evals[-1]],[0,0])
    plt.ylim(-lim,lim)
    
    plt.show()
    
    
    qc=np.array(qc)
    qc[np.abs(qc)>3]=float('NaN')
    qca=np.array(qca)
    qca[np.abs(qca)>40]=float('NaN')
#    plt.plot(qtok(Etoq(Evals,0),mL)**2,qc)
#    plt.plot(qtok(Etoq(Evals,0),mL)**2,qca)
    plt.plot(Evals,qc)
    plt.plot(Evals,qca)
    
    plt.plot([Evals[0],Evals[-1]],[0,0])
    plt.ylim(-0.5,1.5)
    
    plt.show()
    print(getzeros(qca,Evals,40))


#%% Calculate and plot the enrgy levels

#get the energy levels
if(False):
    lam=-100
    mL=5
    As=np.arange(0,0.31,0.01)
    solutions=[]
    for a in As:
        print(a)
        
        #get the rough solutions
        Evals=np.arange(1.75,5,0.00053)
        
        f00=[]
        f00a=[]
        
        qc=[]
        qca=[]      
        
        for e in Evals:            
            qc=qc+[getQC(e,0,lam,mL)]
            qca=qca+[getQC(e,a,lam,mL)]               
        qc=np.array(qc)
        qc[np.abs(qc)>3]=float('NaN')
        qca=np.array(qca)
        qca[np.abs(qca)>200]=float('NaN') 
        tempsol=getzeros(qca,Evals,200)
        print(tempsol)
        
        tempsol=tempsol[0:4]
        solutions=solutions+[tempsol]

    solutions=np.array(solutions)
    solutions=np.column_stack((As,solutions))
    np.savetxt('mL_5_lam_m100.txt',solutions)

if(False):
    
    vals=np.loadtxt('mL_5_lam_m100.txt')
    As=vals[:,0]
    E0=vals[:,1]
    E1=vals[:,2]
    E2=vals[:,3]
    E3=vals[:,4]
    
    E0a0=np.full(len(As),E0[0])
    E1a0=np.full(len(As),E1[0])
    E2a0=np.full(len(As),E2[0])
    E3a0=np.full(len(As),E3[0])
    
    plt.plot(As,E0)
    plt.plot(As,E1)
    plt.plot(As,E2)
    plt.plot(As,E3)
    
    plt.plot(As,E0a0,'C0:')
    plt.plot(As,E1a0,'C1:')
    plt.plot(As,E2a0,'C2:')
    plt.plot(As,E3a0,'C3:')

    plt.xlim(0,0.3)
    plt.ylim(1.8,5)
    plt.show()
    
if(True):
     
    vals=np.loadtxt('mL_5_lam_100.txt')
    As=vals[:,0]
    E0=vals[:,1]
    E1=vals[:,2]
    E2=vals[:,3]
    E3=vals[:,4]
    

    
    plt.plot(As,E0)
    plt.plot(As,E1)
    plt.plot(As,E2)
    plt.plot(As,E3)
    
     
    vals=np.loadtxt('mL_5_lam_100.txt')
    Asold=vals[:,0]
    E0old=vals[:,1]
    E1old=vals[:,2]
    E2old=vals[:,3]
    E3old=vals[:,4]
    
    
    plt.plot(Asold,E0old,'C0--')
    plt.plot(Asold,E1old,'C1--')
    plt.plot(Asold,E2old,'C2--')
    plt.plot(Asold,E3old,'C3--')
    
    plt.xlim(0,0.2)
    plt.ylim(1.8,5)
    plt.show()
    
    E0params, pcov=optimize.curve_fit(afit,Asold,E0old)
    E1params, pcov=optimize.curve_fit(afit,Asold,E1old)
    E2params, pcov=optimize.curve_fit(afit,Asold,E2old)
    E3params, pcov=optimize.curve_fit(afit,Asold,E3old)
    
    plt.plot(As,(E0-afit(As,*E0params))/As**2,'C0:')
    plt.plot(As,(E1-afit(As,*E1params))/As**2,'C1:')
    plt.plot(As,(E2-afit(As,*E2params))/As**2,'C2:')
    plt.plot(As,(E3-afit(As,*E3params))/As**2,'C3:')    
    
    plt.xlim(0,0.3)
    plt.ylim(0,20)
    plt.show()
