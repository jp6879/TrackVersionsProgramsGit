import numpy as np
#import ipdb

class Free_M_NOGSE:
	def __str__(self):
		return 'Free Dif. (NOGSE)'
	def __call__(self,x,amp,D0):
		x=x/1000.
		return amp*np.exp(-self.Gamma**2*self.G**2*D0*((self.N-1)*x**3+(self.T-(self.N-1)*x)**3)/12)
	def __init__(self):
		self.Gamma = 26.7035
		self.param_names=('T','G','N')
		self.param_types=('float','float','int')
		self.param_suffix=('ms','gauss/cm','int')
		self.var_names=('Amp','D0')
		self.var_suffix=('','μm^2/ms')
	def assign_params(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.param_names[i], arguments[i])
		self.G=self.G/10000. #gauss/um
	def assign_seeds(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.var_names[i], arguments[i])
		self.seed = arguments

class Single_Size_NOGSE:
	def __str__(self):
		return 'Single Size (NOGSE)'
	
	def __call__(self,x,amp,lc):
		x=x/1000.
		tc= (lc**2)/(2*self.D0)
		return amp*self.CPMGm((self.N-1)*x,self.N-1,tc)*self.HAHNm((self.T-(self.N-1)*x),tc)*self.NOGSEmcross(x,tc)
	
	def __init__(self):
		self.Gamma = 26.7035
		self.param_names=('T','G','N','D0')
		self.param_types=('float','float','int','float')
		self.param_suffix=('ms','g/cm','int','μm^2/ms')
		self.var_names=('Amp','lc')
		self.var_suffix=('','um')
	
	def assign_params(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.param_names[i], arguments[i])
		self.G=self.G/10000. #gauss/um
	
	def assign_seeds(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.var_names[i], arguments[i])
		self.seed = arguments
	
	def HAHNm(self,TH,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*TH*(1-(tc/TH)*(3+np.exp(-TH/tc)-4*np.exp(-TH/(2*tc)))))
	
	def A(self,TC,NC,tc):
		return (2*NC+1)-np.power(-1,NC)*np.exp(-TC/tc)
	
	def B(self,TC,NC,tc):
		return -((2/(np.exp(-TC/(NC*tc))+1))**2)*(np.power(-1,(NC+1))*np.exp(-TC/tc)*(np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))-np.exp(-TC/(NC*tc)))+np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))+NC*np.exp(-2*TC/(NC*tc))+(NC-1)*np.exp(-TC/(NC*tc)))
	
	def CPMGm(self,TC,NC,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*(TC-tc*(self.A(TC,NC,tc)+self.B(TC,NC,tc))))
	
	def NOGSEmcross(self,x,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**3*((1+np.exp(-(self.T-(self.N-1)*x)/tc)-2*np.exp(-(self.T-(self.N-1)*x)/(2*tc))-2*np.exp((2*x-(self.T-(self.N-1)*x))/(2*tc))+np.exp((x-(self.T-(self.N-1)*x))/tc)+4*np.exp((x-(self.T-(self.N-1)*x))/(2*tc))-2*np.exp((x-2*(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(x/(2*tc))+np.exp(x/tc))+np.power(-1,self.N)*(np.exp(-(x*self.N-x+(self.T-(self.N-1)*x))/tc)-2*np.exp(-(2*x*self.N-2*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-4*x+(self.T-(self.N-1)*x))/(2*tc))+np.exp(-(x*self.N-2*x+(self.T-(self.N-1)*x))/tc)+4*np.exp(-(2*x*self.N-3*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-3*x+2*(self.T-(self.N-1)*x))/(2*tc))+np.exp(-((self.N-1)*x)/tc)-2*np.exp(-((2*self.N-3)*x)/(2*tc))+np.exp(-((self.N-2)*x)/tc)))/(np.exp(x/tc)+1))


class Size_Distribution:
	def __str__(self):
		return 'Size Distribution (NOGSE)'
	
	def __call__(self,x,amp,lc,sigma):
		x=x/1000.
		Lista_l = np.linspace(1,50,51)
		LogNormalp = 0
		NOGSEsp = 0
		for i in Lista_l:
			LogNormalp = LogNormalp + self.LogNormal(i,lc,sigma)
			NOGSEsp = NOGSEsp + self.NOGSEm(x,i)*self.LogNormal(i,lc,sigma)
		return amp*NOGSEsp/LogNormalp
	
	def __init__(self):
		self.Gamma = 26.7035
		self.param_names=('T','G','N','D0')
		self.param_types=('float','float','int','float')
		self.param_suffix=('ms','g/cm','int','μm^2/ms')
		self.var_names=('Amp','lc','σc')
		self.var_suffix=('','um','um')
	def assign_params(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.param_names[i], arguments[i])
		self.G=self.G/10000. #gauss/um
	def assign_seeds(self,arguments):
		for i in range(len(arguments)):
			setattr(self, self.var_names[i], arguments[i])
		self.seed = arguments
	def HAHNm(self,TH,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*TH*(1-(tc/TH)*(3+np.exp(-TH/tc)-4*np.exp(-TH/(2*tc)))))
	def A(self,TC,NC,tc):
		return (2*NC+1)-np.power(-1,NC)*np.exp(-TC/tc)
	def B(self,TC,NC,tc):
		return -((2/(np.exp(-TC/(NC*tc))+1))**2)*(np.power(-1,(NC+1))*np.exp(-TC/tc)*(np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))-np.exp(-TC/(NC*tc)))+np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))+NC*np.exp(-2*TC/(NC*tc))+(NC-1)*np.exp(-TC/(NC*tc)))
	def CPMGm(self,TC,NC,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*(TC-tc*(self.A(TC,NC,tc)+self.B(TC,NC,tc))))
	def NOGSEmcross(self,x,tc):
		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**3*((1+np.exp(-(self.T-(self.N-1)*x)/tc)-2*np.exp(-(self.T-(self.N-1)*x)/(2*tc))-2*np.exp((2*x-(self.T-(self.N-1)*x))/(2*tc))+np.exp((x-(self.T-(self.N-1)*x))/tc)+4*np.exp((x-(self.T-(self.N-1)*x))/(2*tc))-2*np.exp((x-2*(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(x/(2*tc))+np.exp(x/tc))+np.power(-1,self.N)*(np.exp(-(x*self.N-x+(self.T-(self.N-1)*x))/tc)-2*np.exp(-(2*x*self.N-2*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-4*x+(self.T-(self.N-1)*x))/(2*tc))+np.exp(-(x*self.N-2*x+(self.T-(self.N-1)*x))/tc)+4*np.exp(-(2*x*self.N-3*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-3*x+2*(self.T-(self.N-1)*x))/(2*tc))+np.exp(-((self.N-1)*x)/tc)-2*np.exp(-((2*self.N-3)*x)/(2*tc))+np.exp(-((self.N-2)*x)/tc)))/(np.exp(x/tc)+1))
	def NOGSEm(self,x,lc):
		tc= (lc**2)/(2*self.D0)
		return self.CPMGm((self.N-1)*x,self.N-1,tc)*self.HAHNm((self.T-(self.N-1)*x),tc)*self.NOGSEmcross(x,tc)
	def LogNormal(self,l,lc,Sigma):
		return (np.exp((-(np.log(l)-np.log(lc/(np.sqrt(1+(Sigma**2)/(lc**2)))))**2)/(2*np.log(1+(Sigma**2)/(lc**2)))))/(l*np.sqrt(2*np.pi)*np.sqrt(np.log(1+(Sigma**2)/(lc**2))))
	def CPMG_dG(self,T,G,N,x):
		return self.CPMGm((self.N-1)*x,self.N-1,tc)*np.log(self.CPMGm((self.N-1)*x,self.N-1,tc))*2./self.G
	def CPMG_dN(self,T,G,N,x):
		return -D0*G**2*self.Gamma**2*tc**2*(-tc*(-(-1)**(self.N - 1)*1j*np.pi*exp(-x*(self.N - 1)/tc) + (-1)**(self.N - 1)*x*exp(-x*(self.N - 1)/tc)/tc + 2 - 4*((-1)**self.N*((2*x*(self.N - 1)/(tc*(2*self.N - 2)**2) - x/(tc*(2*self.N - 2)))*exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + (6*x*(self.N - 1)/(tc*(2*self.N - 2)**2) - 3*x/(tc*(2*self.N - 2)))*exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))))*exp(-x*(self.N - 1)/tc) + (-1)**self.N*1j*np.pi*(exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))) - exp(-x/tc))*exp(-x*(self.N - 1)/tc) - (-1)**self.N*x*(exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))) - exp(-x/tc))*exp(-x*(self.N - 1)/tc)/tc + (2*x*(self.N - 1)/(tc*(2*self.N - 2)**2) - x/(tc*(2*self.N - 2)))*exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + (6*x*(self.N - 1)/(tc*(2*self.N - 2)**2) - 3*x/(tc*(2*self.N - 2)))*exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))) + exp(-x/tc) + exp(-2*x/tc))/(1 + exp(-x/tc))**2) + x)*exp(-D0*G**2*self.Gamma**2*tc**2*(-tc*(-(-1)**(self.N - 1)*exp(-x*(self.N - 1)/tc) + 2*self.N - 1 - 4*((-1)**self.N*(exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))) - exp(-x/tc))*exp(-x*(self.N - 1)/tc) + (self.N - 2)*exp(-x/tc) + (self.N - 1)*exp(-2*x/tc) + exp(-x*(self.N - 1)/(tc*(2*self.N - 2))) + exp(-3*x*(self.N - 1)/(tc*(2*self.N - 2))))/(1 + exp(-x/tc))**2) + x*(self.N - 1)))










#exp(-Gamma**2*G**2*D0*tc**2*(((N-1)*x)-tc*((2*(N-1)+1)-np.power(-1,(N-1))*exp(-((N-1)*x)/tc)-((2/(exp(-((N-1)*x)/((N-1)*tc))+1))**2)*(np.power(-1,((N-1)+1))*exp(-((N-1)*x)/tc)*(exp(-3*((N-1)*x)/(2*(N-1)*tc))+exp(-((N-1)*x)/(2*(N-1)*tc))-exp(-((N-1)*x)/((N-1)*tc)))+exp(-3*((N-1)*x)/(2*(N-1)*tc))+exp(-((N-1)*x)/(2*(N-1)*tc))+(N-1)*exp(-2*((N-1)*x)/((N-1)*tc))+((N-1)-1)*exp(-((N-1)*x)/((N-1)*tc))))))
#
#
#
#Gamma = Symbol('Gamma')
#G     = Symbol('G')
#D0    = Symbol('D0')
#tc    = Symbol('tc')
#N     = Symbol('N')
#x     = Symbol('x')

#dN
#-D0*G**2*Gamma**2*tc**2*(-tc*(-(-1)**(N - 1)*1j*np.pi*exp(-x*(N - 1)/tc) + (-1)**(N - 1)*x*exp(-x*(N - 1)/tc)/tc + 2 - 4*((-1)**N*((2*x*(N - 1)/(tc*(2*N - 2)**2) - x/(tc*(2*N - 2)))*exp(-x*(N - 1)/(tc*(2*N - 2))) + (6*x*(N - 1)/(tc*(2*N - 2)**2) - 3*x/(tc*(2*N - 2)))*exp(-3*x*(N - 1)/(tc*(2*N - 2))))*exp(-x*(N - 1)/tc) + (-1)**N*1j*np.pi*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc) - (-1)**N*x*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc)/tc + (2*x*(N - 1)/(tc*(2*N - 2)**2) - x/(tc*(2*N - 2)))*exp(-x*(N - 1)/(tc*(2*N - 2))) + (6*x*(N - 1)/(tc*(2*N - 2)**2) - 3*x/(tc*(2*N - 2)))*exp(-3*x*(N - 1)/(tc*(2*N - 2))) + exp(-x/tc) + exp(-2*x/tc))/(1 + exp(-x/tc))**2) + x)*exp(-D0*G**2*Gamma**2*tc**2*(-tc*(-(-1)**(N - 1)*exp(-x*(N - 1)/tc) + 2*N - 1 - 4*((-1)**N*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc) + (N - 2)*exp(-x/tc) + (N - 1)*exp(-2*x/tc) + exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))))/(1 + exp(-x/tc))**2) + x*(N - 1)))

#dx
#-D0*G**2*Gamma**2*tc**2*(N - tc*((-1)**(N - 1)*(N - 1)*exp(-x*(N - 1)/tc)/tc - 4*((-1)**N*(-(N - 1)*exp(-x*(N - 1)/(tc*(2*N - 2)))/(tc*(2*N - 2)) - 3*(N - 1)*exp(-3*x*(N - 1)/(tc*(2*N - 2)))/(tc*(2*N - 2)) + exp(-x/tc)/tc)*exp(-x*(N - 1)/tc) - (-1)**N*(N - 1)*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc)/tc - (N - 2)*exp(-x/tc)/tc - 2*(N - 1)*exp(-2*x/tc)/tc - (N - 1)*exp(-x*(N - 1)/(tc*(2*N - 2)))/(tc*(2*N - 2)) - 3*(N - 1)*exp(-3*x*(N - 1)/(tc*(2*N - 2)))/(tc*(2*N - 2)))/(1 + exp(-x/tc))**2 - 8*((-1)**N*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc) + (N - 2)*exp(-x/tc) + (N - 1)*exp(-2*x/tc) + exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))))*exp(-x/tc)/(tc*(1 + exp(-x/tc))**3)) - 1)*exp(-D0*G**2*Gamma**2*tc**2*(-tc*(-(-1)**(N - 1)*exp(-x*(N - 1)/tc) + 2*N - 1 - 4*((-1)**N*(exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))) - exp(-x/tc))*exp(-x*(N - 1)/tc) + (N - 2)*exp(-x/tc) + (N - 1)*exp(-2*x/tc) + exp(-x*(N - 1)/(tc*(2*N - 2))) + exp(-3*x*(N - 1)/(tc*(2*N - 2))))/(1 + exp(-x/tc))**2) + x*(N - 1)))



#class Fit_Fuxctions:
#	def __init__(self,T,G,N,D0):
#		self.Gamma=26.7035
#		self.T = T
#		self.G = G
#		self.N = N
#		self.D0= D0
#	def HAHNm(self,TH,tc):
#		return exp(-self.Gamma**2*self.G**2*self.D0*tc**2*TH*(1-(tc/TH)*(3+exp(-TH/tc)-4*np.exp(-TH/(2*tc)))))
#	def HAHNr(self,TH,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*(TH-3*tc))
#	def HAHNl(self,TH):
#		return np.exp(-(self.Gamma**2*self.G**2*self.D0*TH**3)/12)
#	def A(self,TC,NC,tc):
#		return (2*NC+1)-np.power(-1,NC)*np.exp(-TC/tc)
#	def B(self,TC,NC,tc):
#		return -((2/(np.exp(-TC/(NC*tc))+1))**2)*(np.power(-1,(NC+1))*np.exp(-TC/tc)*(np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))-np.exp(-TC/(NC*tc)))+np.exp(-3*TC/(2*NC*tc))+np.exp(-TC/(2*NC*tc))+NC*np.exp(-2*TC/(NC*tc))+(NC-1)*np.exp(-TC/(NC*tc)))
#	def CPMGm(self,TC,NC,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*(TC-tc*(self.A(TC,NC,tc)+self.B(TC,NC,tc))))
#	def Ar(self,TC,NC,tc):
#		return (2*NC+1)	
#	def Br(self,TC,NC,tc):
#		return 0.
#	def CPMGr(self,TC,NC,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**2*(TC-tc*(self.Ar(TC,NC,tc)+self.Br(TC,NC,tc))))
#	def CPMGl(self,TC,NC,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*TC**3/(12*NC**2))
#	def NOGSEmcross(self,x,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**3*((1+np.exp(-(self.T-(self.N-1)*x)/tc)-2*np.exp(-(self.T-(self.N-1)*x)/(2*tc))-2*np.exp((2*x-(self.T-(self.N-1)*x))/(2*tc))+np.exp((x-(self.T-(self.N-1)*x))/tc)+4*np.exp((x-(self.T-(self.N-1)*x))/(2*tc))-2*np.exp((x-2*(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(x/(2*tc))+np.exp(x/tc))+np.power(-1,self.N)*(np.exp(-(x*self.N-x+(self.T-(self.N-1)*x))/tc)-2*np.exp(-(2*x*self.N-2*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-4*x+(self.T-(self.N-1)*x))/(2*tc))+np.exp(-(x*self.N-2*x+(self.T-(self.N-1)*x))/tc)+4*np.exp(-(2*x*self.N-3*x+(self.T-(self.N-1)*x))/(2*tc))-2*np.exp(-(2*x*self.N-3*x+2*(self.T-(self.N-1)*x))/(2*tc))+np.exp(-((self.N-1)*x)/tc)-2*np.exp(-((2*self.N-3)*x)/(2*tc))+np.exp(-((self.N-2)*x)/tc)))/(np.exp(x/tc)+1))
#	def NOGSEm(self,x,lc):
#		tc= (lc**2)/(2*self.D0)
#		return self.CPMGm((self.N-1)*x,self.N-1,tc)*self.HAHNm((self.T-(self.N-1)*x),tc)*self.NOGSEmcross(x,tc)
#	def NOGSEmcrossrxy(self,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**3)
#	def NOGSErxy(self,T,N,tc):
#		return self.CPMGl(T,N,tc)
#	def NOGSEmcrossy(self,x,tc):
#		return np.exp(-self.Gamma**2*self.G**2*self.D0*tc**3*(1-(2*np.exp(-x/(2*tc)))/(1+np.exp(-x/tc)))*(1+np.power(-1,self.N)*np.exp(-(self.N-1)*x/tc)))
#	def NOGSEry(self,x,tc):
#		return self.HAHNr((self.T-(self.N-1)*x),tc)*self.CPMGm((self.N-1)*x,(self.N-1),tc)*self.NOGSEmcrossry(x,tc)
#	def NOGSEm_Ajuste(self,x,lc,Amp):
#		return Amp*self.NOGSEm(x,lc)
#	def NOGSEl(self,x,D0):
#		return np.exp(-self.Gamma**2*self.G**2*D0*((self.N-1)*x**3+(self.T-(self.N-1)*x)**3)/12)
#	def NOGSEl_Ajuste(self,x,D0,Amp):
#		return Amp*self.NOGSEl(x,D0)
#	def LogNormal(self,l,lc,Sigma):
#		return (np.exp((-(np.log(l)-np.log(lc/(np.sqrt(1+(Sigma**2)/(lc**2)))))**2)/(2*np.log(1+(Sigma**2)/(lc**2)))))/(l*np.sqrt(2*np.pi)*np.sqrt(np.log(1+(Sigma**2)/(lc**2))))
#	def NOGSEs(self,x,lc,Sigma):
#		Lista_l = np.linspace(1,50,51)
#		LogNormalp = 0
#		NOGSEsp = 0
#		for i in Lista_l:
#		    LogNormalp = LogNormalp + self.LogNormal(i,lc,Sigma)
#		    NOGSEsp = NOGSEsp + self.NOGSEm(x,i)*self.LogNormal(i,lc,Sigma)
#		return NOGSEsp/LogNormalp
#	def NOGSEs_Ajuste(self,x,lc,Sigma,Amp):
#		return Amp*self.NOGSEs(x,lc,Sigma)