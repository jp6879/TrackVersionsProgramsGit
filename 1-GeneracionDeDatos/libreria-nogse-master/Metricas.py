import numpy as np
from scipy import integrate

def filtro(T,G,N,nogse,sizes): #nogse puede ser de distribucion o de tamaño unico. El resultado es el mismo
	return nogse.NOGSEm(T/N,T,G,N,sizes)-nogse.NOGSEm(0,T,G,N,sizes)

def filtro_distrib(T,G,N,nogse,distrib,axis=None): #nogse debe ser de distribucion. distrib=(amp,mu,s)
	return nogse.signal(T/N,T,G,N,dis=distrib,axis=axis)-nogse.signal(0,T,G,N,dis=distrib,axis=axis)

def distancia(x,T,G,N,nogse,distrib1,distrib_real,axis=1):
	aux = np.square( nogse.signal(x,T,G,N,dis=distrib1,axis=axis) -  nogse.signal(x,T,G,N,dis=distrib_real,axis=axis) )
	return np.sqrt(integrate.simps(aux,x.T[0].T))

def svty(x,T,G,N,nogse,distrib,delta1=0.1,delta2=0.1,n_angles=128):
	thetas = np.linspace(0.,2.*np.pi,n_angles)
	mus = distrib[1]*(1. + delta1*np.cos(thetas))[:,None,None]
	sigs = distrib[2]*(1. + delta2*np.sin(thetas))[:,None,None]
	return distancia(x,T,G,N,nogse,(distrib[0],mus,sigs),distrib,axis=2)