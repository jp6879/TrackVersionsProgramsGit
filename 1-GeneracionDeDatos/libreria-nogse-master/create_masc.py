import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector 
from shapely.geometry import Point,Polygon  
plt.ion()

cmap=plt.cm.jet
cmap.set_bad('black')

def func(img,pol): 
		masc = np.ones(img.shape) 
		for i in range(img.shape[0]):  
			for j in range(img.shape[1]):  
				if pol.contains(Point([i,j]))==False:  
					masc[i,j]=np.nan 
		return masc
class onselect:
	def __init__(self):
		self.vert = []
	def __call__(self,verts):
		self.vert.append([[f[1],f[0]] for f in verts])

def create_masc(img):
	fig, ax = plt.subplots() 
	ax.imshow(img, cmap=cmap) 
	return ax
#lasso = LassoSelector(ax, onselect) 
#pol = Polygon(vert[0])

def mask(vert,img):
	pol = Polygon(vert[0])
	return func(img,pol)
