from scipy.special import erf
import numpy as np

def lognorm_primitive(l,mu,s):
	return -0.5*erf((mu-np.log(l))/(np.sqrt(2)*s))

class Magnetization:
	def __init__(self,D0):
		self.D0=D0
		self.Gamma = 26.7035
	def HAHNm(self,TH,G,l):
		tc= (l**2)/(2*self.D0)
		return np.exp(-self.Gamma**2*G**2*self.D0*tc**2*TH*(1-(tc/TH)*(3+np.exp(-TH/tc)-4*np.exp(-TH/(2*tc)))))
	def A(self,TC,NC,tc):
		return (2*NC+1)-np.power(-1,NC)*np.exp(-TC/tc)
	def B(self,TC,NC,tc):
		return -((2/(np.exp(-TC/(NC*tc))+1))**2)*(np.power(-1,(NC+1))*np.exp(-TC/tc)*(np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))-np.exp(-TC/(NC*tc)))+np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))+NC*np.exp(-2*TC/(NC*tc))+(NC-1)*np.exp(-TC/(NC*tc)))
	def CPMGm(self,TC,NC,G,l):
		tc= (l**2)/(2*self.D0)
		return np.exp(-self.Gamma**2*G**2*self.D0*tc**2*(TC-tc*(self.A(TC,NC,tc)+self.B(TC,NC,tc))))
	def NOGSEmcross(self,x,T,G,N,l):
		tc= (l**2)/(2*self.D0)
		return np.exp(-self.Gamma**2*G**2*self.D0*tc**3*((1+np.exp(-(T-(N-1)*x)/tc)-2*np.exp(-(T-(N-1)*x)/(2*tc))-2*np.exp((2*x-(T-(N-1)*x))/(2*tc))+np.exp((x-(T-(N-1)*x))/tc)+4*np.exp((x-(T-(N-1)*x))/(2*tc))-2*np.exp((x-2*(T-(N-1)*x))/(2*tc))-2*np.exp(x/(2*tc))+np.exp(x/tc))+np.power(-1,N)*(np.exp(-(x*N-x+(T-(N-1)*x))/tc)-2*np.exp(-(2*x*N-2*x+(T-(N-1)*x))/(2*tc))-2*np.exp(-(2*x*N-4*x+(T-(N-1)*x))/(2*tc))+np.exp(-(x*N-2*x+(T-(N-1)*x))/tc)+4*np.exp(-(2*x*N-3*x+(T-(N-1)*x))/(2*tc))-2*np.exp(-(2*x*N-3*x+2*(T-(N-1)*x))/(2*tc))+np.exp(-((N-1)*x)/tc)-2*np.exp(-((2*N-3)*x)/(2*tc))+np.exp(-((N-2)*x)/tc)))/(np.exp(x/tc)+1))
	def NOGSEm(self,x,T,G,N,l):
		return self.CPMGm((N-1)*x,N-1,G,l)*self.HAHNm((T-(N-1)*x),G,l)*self.NOGSEmcross(x,T,G,N,l)
	def LogNormal(self,l,lc,Sigma):
		return np.exp(-(np.log(l)-lc)**2/(2.*Sigma**2))/(l*Sigma*np.sqrt(2.*np.pi))
	def dCPMGdG(self,x,T,G,N,l):
		return -G*self.Gamma**2*l**4*(x*(N - 1) - l**2*(-(-1)**(N - 1)*np.exp(2*self.D0*x*(1 - N)/l**2) + 2*N - 1 - 4*((-1)**N*(np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) - np.exp(-2*self.D0*x/l**2))*np.exp(-2*self.D0*x*(N - 1)/l**2) + (N - 2)*np.exp(-2*self.D0*x/l**2) + (N - 1)*np.exp(-4*self.D0*x/l**2) + np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))))/(1 + np.exp(-2*self.D0*x/l**2))**2)/(2*self.D0))*np.exp(-G**2*self.Gamma**2*l**4*(x*(N - 1) - l**2*(-(-1)**(N - 1)*np.exp(2*self.D0*x*(1 - N)/l**2) + 2*N - 1 - 4*((-1)**N*(np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) - np.exp(-2*self.D0*x/l**2))*np.exp(-2*self.D0*x*(N - 1)/l**2) + (N - 2)*np.exp(-2*self.D0*x/l**2) + (N - 1)*np.exp(-4*self.D0*x/l**2) + np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))))/(1 + np.exp(-2*self.D0*x/l**2))**2)/(2*self.D0))/(4*self.D0))/(2*self.D0)
	def dCPMGdN(self,x,T,G,N,l):
		return -G**2*self.Gamma**2*l**4*(x - l**2*(2*(-1)**(N - 1)*self.D0*x*np.exp(2*self.D0*x*(1 - N)/l**2)/l**2 - (-1)**(N - 1)*np.exp(2*self.D0*x*(1 - N)/l**2) + 2 - 4*(-2*(-1)**N*self.D0*x*(np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) - np.exp(-2*self.D0*x/l**2))*np.exp(-2*self.D0*x*(N - 1)/l**2)/l**2 + (-1)**N*((4*self.D0*x*(N - 1)/(l**2*(2*N - 2)**2) - 2*self.D0*x/(l**2*(2*N - 2)))*np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + (12*self.D0*x*(N - 1)/(l**2*(2*N - 2)**2) - 6*self.D0*x/(l**2*(2*N - 2)))*np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))))*np.exp(-2*self.D0*x*(N - 1)/l**2) + (-1)**N*(np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) - np.exp(-2*self.D0*x/l**2))*np.exp(-2*self.D0*x*(N - 1)/l**2) + (4*self.D0*x*(N - 1)/(l**2*(2*N - 2)**2) - 2*self.D0*x/(l**2*(2*N - 2)))*np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + (12*self.D0*x*(N - 1)/(l**2*(2*N - 2)**2) - 6*self.D0*x/(l**2*(2*N - 2)))*np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-2*self.D0*x/l**2) + np.exp(-4*self.D0*x/l**2))/(1 + np.exp(-2*self.D0*x/l**2))**2)/(2*self.D0))*np.exp(-G**2*self.Gamma**2*l**4*(x*(N - 1) - l**2*(-(-1)**(N - 1)*np.exp(2*self.D0*x*(1 - N)/l**2) + 2*N - 1 - 4*((-1)**N*(np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))) - np.exp(-2*self.D0*x/l**2))*np.exp(-2*self.D0*x*(N - 1)/l**2) + (N - 2)*np.exp(-2*self.D0*x/l**2) + (N - 1)*np.exp(-4*self.D0*x/l**2) + np.exp(-2*self.D0*x*(N - 1)/(l**2*(2*N - 2))) + np.exp(-6*self.D0*x*(N - 1)/(l**2*(2*N - 2))))/(1 + np.exp(-2*self.D0*x/l**2))**2)/(2*self.D0))/(4*self.D0))/(4*self.D0)
	def dHAHNdG(self,x,T,G,N,l):
		return -G*self.Gamma**2*l**4*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))*(T - x*(N - 1))*np.exp(-G**2*self.Gamma**2*l**4*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))*(T - x*(N - 1))/(4*self.D0))/(2*self.D0)
	def dHAHNdN(self,x,T,G,N,l):
		return (G**2*self.Gamma**2*l**4*x*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))/(4*self.D0) - G**2*self.Gamma**2*l**4*(T - x*(N - 1))*(-l**2*x*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))**2) - l**2*(2*self.D0*x*np.exp(2*self.D0*(-T + x*(N - 1))/l**2)/l**2 - 4*self.D0*x*np.exp(self.D0*(-T + x*(N - 1))/l**2)/l**2)/(2*self.D0*(T - x*(N - 1))))/(4*self.D0))*np.exp(-G**2*self.Gamma**2*l**4*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))*(T - x*(N - 1))/(4*self.D0))
	def dHAHNdT(self,x,T,G,N,l):
		return (-G**2*self.Gamma**2*l**4*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))/(4*self.D0) - G**2*self.Gamma**2*l**4*(T - x*(N - 1))*(-l**2*(-2*self.D0*np.exp(2*self.D0*(-T + x*(N - 1))/l**2)/l**2 + 4*self.D0*np.exp(self.D0*(-T + x*(N - 1))/l**2)/l**2)/(2*self.D0*(T - x*(N - 1))) + l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))**2))/(4*self.D0))*np.exp(-G**2*self.Gamma**2*l**4*(1 - l**2*(np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 4*np.exp(self.D0*(-T + x*(N - 1))/l**2) + 3)/(2*self.D0*(T - x*(N - 1))))*(T - x*(N - 1))/(4*self.D0))
	def dCROSSdT(self,x,T,G,N,l):
		return -G**2*self.Gamma**2*l**6*((-1)**N*(4*self.D0*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2)/l**2 + 2*self.D0*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2)/l**2 - 4*self.D0*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2)/l**2 + 2*self.D0*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2)/l**2 - 2*self.D0*np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2)/l**2 - 2*self.D0*np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2)/l**2) - 2*self.D0*np.exp(2*self.D0*(-T + x*(N - 1))/l**2)/l**2 + 2*self.D0*np.exp(self.D0*(-T + x*(N - 1))/l**2)/l**2 + 4*self.D0*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2)/l**2 - 2*self.D0*np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2)/l**2 - 4*self.D0*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2)/l**2 + 2*self.D0*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2)/l**2)*np.exp(-G**2*self.Gamma**2*l**6*((-1)**N*(-2*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2) + 4*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2) - 2*np.exp(-self.D0*x*(2*N - 3)/l**2) + np.exp(-2*self.D0*x*(N - 1)/l**2) + np.exp(-2*self.D0*x*(N - 2)/l**2)) + np.exp(2*self.D0*x/l**2) - 2*np.exp(self.D0*x/l**2) + np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2) + 4*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2) + 1)/(8*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1)))/(8*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1))
	def dCROSSdG(self,x,T,G,N,l):
		return -G*self.Gamma**2*l**6*((-1)**N*(-2*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2) + 4*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2) - 2*np.exp(-self.D0*x*(2*N - 3)/l**2) + np.exp(-2*self.D0*x*(N - 1)/l**2) + np.exp(-2*self.D0*x*(N - 2)/l**2)) + np.exp(2*self.D0*x/l**2) - 2*np.exp(self.D0*x/l**2) + np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2) + 4*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2) + 1)*np.exp(-G**2*self.Gamma**2*l**6*((-1)**N*(-2*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2) + 4*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2) - 2*np.exp(-self.D0*x*(2*N - 3)/l**2) + np.exp(-2*self.D0*x*(N - 1)/l**2) + np.exp(-2*self.D0*x*(N - 2)/l**2)) + np.exp(2*self.D0*x/l**2) - 2*np.exp(self.D0*x/l**2) + np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2) + 4*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2) + 1)/(8*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1)))/(4*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1))
	def dCROSSdN(self,x,T,G,N,l):
		return -G**2*self.Gamma**2*l**6*((-1)**N*(2*self.D0*x*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2)/l**2 - 4*self.D0*x*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2)/l**2 + 2*self.D0*x*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2)/l**2 + 4*self.D0*x*np.exp(-self.D0*x*(2*N - 3)/l**2)/l**2 - 2*self.D0*x*np.exp(-2*self.D0*x*(N - 1)/l**2)/l**2 - 2*self.D0*x*np.exp(-2*self.D0*x*(N - 2)/l**2)/l**2) + (-1)**N*(-2*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2) + 4*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2) - 2*np.exp(-self.D0*x*(2*N - 3)/l**2) + np.exp(-2*self.D0*x*(N - 1)/l**2) + np.exp(-2*self.D0*x*(N - 2)/l**2)) + 2*self.D0*x*np.exp(2*self.D0*(-T + x*(N - 1))/l**2)/l**2 - 2*self.D0*x*np.exp(self.D0*(-T + x*(N - 1))/l**2)/l**2 - 4*self.D0*x*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2)/l**2 + 2*self.D0*x*np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2)/l**2 + 4*self.D0*x*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2)/l**2 - 2*self.D0*x*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2)/l**2)*np.exp(-G**2*self.Gamma**2*l**6*((-1)**N*(-2*np.exp(self.D0*(-2*N*x - 2*T + 2*x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 2*x)/l**2) + 4*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 3*x)/l**2) - 2*np.exp(self.D0*(-2*N*x - T + x*(N - 1) + 4*x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-N*x - T + x*(N - 1) + 2*x)/l**2) - 2*np.exp(-self.D0*x*(2*N - 3)/l**2) + np.exp(-2*self.D0*x*(N - 1)/l**2) + np.exp(-2*self.D0*x*(N - 2)/l**2)) + np.exp(2*self.D0*x/l**2) - 2*np.exp(self.D0*x/l**2) + np.exp(2*self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1))/l**2) - 2*np.exp(self.D0*(-2*T + 2*x*(N - 1) + x)/l**2) + np.exp(2*self.D0*(-T + x*(N - 1) + x)/l**2) + 4*np.exp(self.D0*(-T + x*(N - 1) + x)/l**2) - 2*np.exp(self.D0*(-T + x*(N - 1) + 2*x)/l**2) + 1)/(8*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1)))/(8*self.D0**2*(np.exp(2*self.D0*x/l**2) + 1))
	def dNOGSEmdG(self,x,T,G,N,l):
		tc= (l**2)/(2*self.D0)
		cpmg = self.CPMGm((N-1)*x,N-1,tc,G)
		hahn = self.HAHNm((T-(N-1)*x),tc,G)
		cross = self.NOGSEmcross(x,tc,T,G,N)
		return self.dCPMGdG(x,l,T,G,N)*hahn*cross+cpmg*self.dHAHNdG(x,l,T,G,N)*cross+cpmg*hahn*self.dCROSSdG(x,tc,T,G,N)
	def dNOGSEmdN(self,x,T,G,N,l):
		tc= (l**2)/(2*self.D0)
		cpmg = self.CPMGm((N-1)*x,N-1,tc,G)
		hahn = self.HAHNm((T-(N-1)*x),tc,G)
		cross = self.NOGSEmcross(x,tc,T,G,N)
		return self.dCPMGdN(x,l,T,G,N)*hahn*cross+cpmg*self.dHAHNdN(x,l,T,G,N)*cross+cpmg*hahn*self.dCROSSdN(x,tc,T,G,N)
	def dNOGSEmdT(self,x,T,G,N,l):
		tc= (l**2)/(2*self.D0)
		cpmg = self.CPMGm((N-1)*x,N-1,tc,G)
		hahn = self.HAHNm((T-(N-1)*x),tc,G)
		cross = self.NOGSEmcross(x,tc,T,G,N)
		return cpmg*self.dHAHNdT(x,l,T,G,N)*cross+cpmg*hahn*self.dCROSSdT(x,tc,T,G,N)
	def NOGSEm_w_free(self,x,T,G,N,l,alpha):
		return self.NOGSEm(x,T,G,N,l)+alpha*np.exp(-self.Gamma**2*G**2*self.D0*((N-1)*x**3+(T-(N-1)*x)**3)/12)


class Distribution_Signal(Magnetization):
	def __init__(self,D0,sizes): #los tamaños se suponen equiespaciados E IMPARES
		super().__init__(D0)
		self.sizes = sizes
		self.weights = np.ones(sizes.shape) #pesos para la integracion por simpson
		if(sizes.shape[0]%2==0):
			print("La cantidad de tamaños debe ser impar")
		dl = (sizes[-1]-sizes[0])/(sizes.shape[0]-1)
		self.weights[1::2]*=4.*dl/3.;self.weights[2:-1:2]*=2.*dl/3.;self.weights[::sizes.shape[0]-1]*=dl/3
		self.weights_signal = self.weights

	def create_size_signal(self,*args,function="NOGSEm"):
		self.weights_signal = getattr(self, function)(*args,self.sizes)*self.weights
	
	def signal(self,*args,dis=(1.,1.,0.1),function="NOGSEm",axis=None):
		amp= dis[0]
		mu = dis[1]
		s  = dis[2]
		self.create_size_signal(*args,function=function)
		aux = self.LogNormal(self.sizes,mu,s)
		primitive = (lognorm_primitive(self.sizes[-1],mu,s)-lognorm_primitive(self.sizes[0],mu,s))
		if np.isscalar(primitive)==False:
			primitive = primitive.T[0].T
		return (amp*self.weights_signal*aux).sum(axis=axis)/primitive
	
	def signal_fit(self,idx,amp,mu,s,axis=None):
		aux = self.LogNormal(self.sizes,mu,s)
		if np.isscalar(idx):
			return amp*(self.weights_signal[idx]*aux).sum(axis=axis)/(lognorm_primitive(self.sizes[-1],mu,s)-lognorm_primitive(self.sizes[0],mu,s))
		return amp*(self.weights_signal[idx.astype(int)]*aux).sum(axis=axis)/(lognorm_primitive(self.sizes[-1],mu,s)-lognorm_primitive(self.sizes[0],mu,s))