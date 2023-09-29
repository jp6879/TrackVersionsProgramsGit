import numpy as np
import os
import json

class PV_Parameters:
	def __init__(self,path):
		self.path = path
		self.obj = add_to_json(path+'/method')
		self.obj.update(add_to_json(path+'/acqp'))
		self.sequence_type=self.obj["Method"][6:-1] 
	def __str__(self):
		try:
			return ('<Experimento '+str(self.folder)+'. Secuencia '+self.sequence_type+'>')
		except: 
			return '<Secuencia vacía>'

def add_to_json(file):
	taken = open(file,'r').read().split('\n')
	taken = list(filter(lambda h: h[0:2]!='$$',taken))	
	x = '{'
	startmatrix = False
	for line in taken:
		if len(line)==0:
			continue
		if line[0].isdigit() or line[0]=='-':
			if startmatrix==False:
				value=line.replace(",","").split()
				if variable_for_json(value)==False:
					continue
				x=x[:x.rfind(':')]+': '
				x=x+'['+variable_for_json(value)+',\n'
				startmatrix=True
				continue
			else:
				value = line.split()
				x=x[:-1]+variable_for_json(value)+',\n'
				continue
			startmatrix=True
			continue
		if line[0]=='<':
			if startmatrix==False:
				x=x[:x.rfind(':')]+': ["'+line+'",\n'
				startmatrix=True
			else:
				x=x+'\"'+line+'",\n'
			continue
		if '##$' in line:
			if startmatrix:
				startmatrix=False
				x=x[:-2]+'],\n'
			x=x+'"'+line[line.find('$')+len('$'):line.rfind('=')]+'": '
			value = line[line.find('=')+len('='):]
			x=x+variable_for_json(value)+',\n'
			continue
	return json.loads(x[:-2]+'}')
def varl(val):
	try:
		if '.' in val:
			x=float(val)
		else:
			x=int(val)
	except:
		x='\"'+str(val)+'\"'
	return x
def variable_for_json(value): #value=string
	if len(value)==0:
		return '\"Empty\"'
	if type(value)!= str:
		var=[]
		for i in range(len(value)):
			value[i]=value[i].replace(',','')
			value[i]=value[i].replace(')','')
			aux = varl(value[i])
			var.append(aux)
		if len(var)==0:
			return '\"Empty\"'
		return str(var).replace("\'","")
	if '(' in value:
		var=[]
		value = value[value.find('(')+len('('):value.rfind(')')].split(',')
		for i in range(len(value)):
			if varl(value[i])==False:
				continue
			aux = varl(value[i])
			var.append(aux)
		return str(var).replace("\'","")
	return str(varl(value))