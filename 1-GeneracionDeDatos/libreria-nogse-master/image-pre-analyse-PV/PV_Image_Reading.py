import numpy as np
#import ipdb

class PV_Images:
	def __init__(self,path):
		self.path = path
		self.parameter_data()
		self.image_set = np.array(np.memmap(self.path+'/pdata/1/2dseq', dtype=np.uint16, shape=(self.nimages,self.shape[1], self.shape[0])))
		#if self.nimages==1:
		#	self.image_set = np.expand_dims(self.image_set,axis=1)
	def parameter_data(self):
		file = self.path+'/pdata/1/visu_pars'
		with open(file,'r') as x:
			lines = x.read().splitlines()
			for i in range(len(lines)):
				if lines[i].startswith('##$VisuCoreFrameCount='):
					self.nimages = int(lines[i][22:])
				if lines[i].startswith('##$VisuCoreSize'):
					self.shape = [int(f) for f in lines[i+1].split(' ')]