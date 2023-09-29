#Edite en pyqtgraph la clase ConsoleWidget
#https://askubuntu.com/questions/138908/how-to-execute-a-script-just-by-double-clicking-like-exe-files-in-windows
#https://askubuntu.com/questions/64222/how-can-i-create-launchers-on-my-desktop



#ORDENAR LISTAS DE EXPERIMENTOS
#Agregar angulos theta y phi

import sys, os 
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

import sys, inspect
from Fit_Functions import *

def print_classes():
    fit_classes=[]
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            fit_classes.append(name)
    return fit_classes
#import ipdb
classes = print_classes()


import datetime
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout,QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import time
import pandas as pd
from functools import partial
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from PyQt5 import QtWidgets
import os
import pyperclip
from PV_Parameter_Reading import *
from PV_Image_Reading import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph.console
from scipy.optimize import curve_fit
import pyqtgraph.opengl as gl

palette = QtGui.QPalette()
palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15,15,15))
palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)


pg.setConfigOptions(imageAxisOrder='row-major')
spam = None
imgss=['./0_0.png']
itemss=['1']

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def load(table):
    index = table.selectedIndexes()
    return [table.model().data(f) for f in index]
class WindowClose(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()
        self.open_win = True
    def closeEvent(self, event):
        self.open_win = False

class Animation:
    def __init__(self):
        global anim_win
        anim_win = WindowClose()
        anim_win.resize(400,400)
        win = pg.GraphicsLayoutWidget()
        anim_win.setCentralWidget(win)
        view = win.addViewBox()
        view.setAspectLocked(True)
        self.anim_img = pg.ImageItem(border='w')
        view.addItem(self.anim_img)
        self.i_update=0
        self.anim_img.setImage(self.images[self.i_update,self.nsubimage])
        QtCore.QTimer.singleShot(1, self.updateData)
        anim_win.show()

class D3Image:
    def __init__(self):
        global d3_win
        d3_win = QtGui.QMainWindow()
        d3_win.resize(400,400)
        w = gl.GLViewWidget()
        d3_win.setCentralWidget(w)
        shape = (self.images[self.nimage,0].shape[0],self.images[self.nimage,0].shape[0],self.images[self.nimage,0].shape[0])
        tex1 = pg.makeRGBA(self.images[self.nimage,0])[0]       # yz plane
        tex2 = pg.makeRGBA(self.images[self.nimage,1])[0]     # xz plane
        tex3 = pg.makeRGBA(self.images[self.nimage,2])[0]   # xy plane
        
        ## Create three image items from textures, add to view
        v1 = gl.GLImageItem(tex1)
        v1.translate(-shape[1]//2, -shape[2]//2, self.sequences[self.nimage].obj['PVM_SliceOffset'][0][0]*6.35)
        v1.rotate(90, 0,0,1)
        v1.rotate(-90, 0,1,0)
        w.addItem(v1)
        v2 = gl.GLImageItem(tex3)
        v2.translate(-shape[0]//2, -shape[2]//2, self.sequences[self.nimage].obj['PVM_SliceOffset'][0][1]*6.35)
        v2.rotate(-90, 1,0,0)
        w.addItem(v2)
        v3 = gl.GLImageItem(tex2)
        v3.translate(-shape[0]//2, -shape[1]//2, self.sequences[self.nimage].obj['PVM_SliceOffset'][0][2]*6.35)
        w.addItem(v3)
        
        ax = gl.GLAxisItem()
        w.addItem(ax)
        d3_win.show()

class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1075, 84)
        self.progressBar = QtWidgets.QProgressBar(Form)
        self.progressBar.setGeometry(QtCore.QRect(30, 30, 1000, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMinimumSize(QtCore.QSize(1000, 35))
        self.progressBar.setMaximumSize(QtCore.QSize(1000, 35))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar") 
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def retranslateUi(self, Form):
      _translate = QtCore.QCoreApplication.translate
      Form.setWindowTitle(_translate("Form", "Progress bar"))


class ProgressBar(QtWidgets.QDialog, Ui_Form):
    def __init__(self, desc = None, parent=None):
        super(ProgressBar, self).__init__(parent)
        self.setupUi(self)
        self.show()
        if desc != None:
            self.setDescription(desc)
    def setValue(self, val): # Sets value
        self.progressBar.setProperty("value", val)
    def setDescription(self, desc): # Sets Pbar window title
        self.setWindowTitle(desc)

times = np.zeros((5))
class Images_Widget:
    def __init__(self):
        global times
        self.variable_list()
        self.seq_parameters()
        self.image_gallery()
        self.pb.setValue(52)
        QApplication.processEvents()
        self.console()
        self.pb.setValue(96)
        QApplication.processEvents()

class Phantom_Widget:
    def __init__(self,img):
        self.colors = [ [255,0,0], [0,255,247],[43,255,0], [255,0,127],[255,255,0],[178,102,255],[204,204,0],[0,0,255],[255,229,204],[255,128,0],[153,255,153]]
        self.phantom_image=img
        if self.created_phantom==False:
            self.phantom_rois = []
            self.phantom_rois.append(pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True,pen=self.colors[0],removable=True))
            self.created_phantom=True
        else:
            for i in range(len(self.phantom_rois)):
                aux = pg.PolyLineROI([np.array([f[1].x(),f[1].y()])+np.array(self.phantom_rois[i].pos()) for f in self.phantom_rois[i].getLocalHandlePositions()], closed=True,pen=self.colors[i],removable=True)
                self.phantom_rois[i]=aux
        self.phantom_w1 = pg.GraphicsView()
        self.phantom_aux= pg.ViewBox(lockAspect=True)
        self.phantom_img1a = pg.ImageItem(self.phantom_image)
        self.phantom_aux.addItem(self.phantom_img1a)
        self.phantom_aux.disableAutoRange('xy')
        self.phantom_aux.autoRange()
        self.phantom_w1.setCentralItem(self.phantom_aux)
        self.layout0.addWidget(self.phantom_w1,0,0,1,1)
        self.phantom_params = [{'name': 'Regions', 'type': 'int', 'value': 1}]
        self.phantom_p = Parameter.create(name='params', type='group', children=self.phantom_params)
        for i in range(len(self.phantom_rois)):
            self.phantom_p.addChild({'name': 'Region '+str(i+1), 'type': 'color', 'value': self.colors[(i)%len(self.colors)],'readonly':True})
        self.phantom_p.sigTreeStateChanged.connect(self.phantom_change)
        self.phantom_t = ParameterTree()
        self.phantom_t.setParameters(self.phantom_p, showTop=False)
        self.phantom_t.setWindowTitle('pyqtgraph example: Parameter Tree')
        self.layout0.addWidget(self.phantom_t,1,0,1,1)
        #vb = pg.ViewBox()
        #vb1 = pg.ViewBox()
        #vb2 = pg.ViewBox()
        #vb3 = pg.ViewBox()
        #vb4 = pg.ViewBox()
        #self.phantom_pw = pg.PlotWidget(viewBox=vb, enableMenu=False, title="Mean Signal")
        #self.phantom_pwwww = pg.PlotWidget(viewBox=vb1, enableMenu=False, title="Region Area")
        #self.phantom_pww = pg.PlotWidget(viewBox=vb2, enableMenu=False, title="Signal Standard Desviation")
        #self.phantom_pwww = pg.PlotWidget(viewBox=vb3, enableMenu=False, title="Region Histogram")  
        #self.phantom_pwwwww = pg.PlotWidget(viewBox=vb4, enableMenu=False, title="Total Signal")   

        self.pwm1 = pg.GraphicsWindow()
        self.phantom_p_w1 = self.pwm1.addPlot(title="Mean Signal")
        self.pwm2 = pg.GraphicsWindow()
        self.phantom_p_w2 = self.pwm2.addPlot(title="Region Area")
        self.pwm3 = pg.GraphicsWindow()
        self.phantom_p_w3 = self.pwm3.addPlot(title="Signal Standard Desviation")
        self.pwm4 = pg.GraphicsWindow()
        self.phantom_p_w4 = self.pwm4.addPlot(title="Total Signal")
        self.pwm5 = pg.GraphicsWindow()
        self.phantom_p_w5 = self.pwm5.addPlot(title="Region Histogram")

        self.phantom_pw1=self.phantom_p_w1.plot()
        self.phantom_pw2=self.phantom_p_w2.plot()
        self.phantom_pw3=self.phantom_p_w3.plot()
        self.phantom_pw4=self.phantom_p_w4.plot()
        self.phantom_pw5 = self.phantom_p_w5.plot(np.zeros((30)),np.zeros((29)),stepMode=True, fillLevel=0, brush=(0,0,255,150))
        
        for roi in self.phantom_rois:
          roi.sigRegionChanged.connect(self.phantom_update)
          roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
          self.phantom_aux.addItem(roi)
          roi.sigClicked.connect(lambda: self.region_histogram(roi))
        medio=np.zeros((len(self.phantom_rois)))
        desv=np.zeros((len(self.phantom_rois)))
        area=np.zeros((len(self.phantom_rois)))
        total=np.zeros((len(self.phantom_rois)))
        for i in range(len(self.phantom_rois)):
          auxiliar=np.array(self.phantom_rois[i].getArrayRegion(self.phantom_image, self.phantom_img1a, axes=(0, 1)))
          medio[i]=auxiliar[auxiliar!=0.].mean()
          desv[i]=auxiliar[auxiliar!=0.].std()
          area[i]=PolygonArea([[f[1].x(),f[1].y()] for f in self.phantom_rois[i].getSceneHandlePositions()])
          total[i]=auxiliar.sum()
        auxiliar=auxiliar.flatten()
        auxiliar=auxiliar[auxiliar!=0.]
        y,x = np.histogram(auxiliar, bins=np.linspace(auxiliar.min(), auxiliar.max(), 30))
        self.phantom_pw5.setData(x, y)
        self.phantom_pw1.setData(x=list(range(len(medio))), y=medio, symbol='o')
        self.phantom_pw2.setData(x=list(range(len(area))), y=area,symbol='o')
        self.phantom_pw3.setData(x=list(range(len(desv))), y=desv,symbol='o')
        self.phantom_pw4.setData(x=list(range(len(total))), y=total,symbol='o')
        self.layout0.addWidget(self.pwm1,0,2,1,1)
        self.layout0.addWidget(self.pwm2,0,3,1,1)
        self.layout0.addWidget(self.pwm3,1,2,1,1)
        self.layout0.addWidget(self.pwm4,0,4,1,1)
        self.layout0.addWidget(self.pwm5,1,3,1,2)

class Model(QAbstractTableModel):
    def __init__(self, parent=None, *args):
        global imgss
        global itemss
        QAbstractTableModel.__init__(self, parent, *args)
        self.images = imgss
        self.items = list(itemss)
        self.thumbSize=100
        self.trig=-1
    def resizePixmap(self, mult):
        self.thumbSize=self.thumbSize*mult
    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable #| Qt.ItemIsEditable

    def rowCount(self, parent):
        return len(self.items)
    def columnCount(self, parent):
        return 2

    def data(self, index, role=Qt.DecorationRole):
        if not index.isValid(): return QVariant()
        row=index.row()
        if row>len(self.items): 
            return QVariant()
        if role == Qt.DisplayRole: #
            return QVariant(self.items[row])

        elif role == Qt.DecorationRole:
          image=self.images[row]
          pixmap=QPixmap(image).scaled(QSize(self.thumbSize, self.thumbSize), Qt.KeepAspectRatio)
          return pixmap
        return QVariant()
    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid():            
            if role == Qt.EditRole:                
                row = index.row()
                self.items[row]=value
                return True
        return False


class ImageMenu(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.tablemodel=Model(self)               
        self.tableviewA=QTableView() 
        self.tableviewA.setModel(self.tablemodel)   
        layout = QVBoxLayout(self)
        layout.addWidget(self.tableviewA)
        self.setLayout(layout)
        #self.tableviewA.resizeColumnToContents(True)
        #self.tableviewA.resizeRowToContents(True)
#        self.tablemodel.resizePixmap(1.)
        thumbSize=self.tableviewA.model().thumbSize 
        totalRows=self.tablemodel.rowCount(QModelIndex())
        for row in range(totalRows):
          self.tableviewA.setRowHeight(row, thumbSize*1)
        self.tableviewA.setColumnHidden(0,True)
        self.tableviewA.resizeRowToContents(True)
        self.tableviewA.resizeColumnToContents(True)




class MyTableWidget(QWidget):
    
    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QVBoxLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab1,"Phantom")
        self.tabs.addTab(self.tab2,"Images")
        self.tabs.addTab(self.tab3,"Graphs")
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

class Search_Window:
    def __init__(self,path):
        global win_open
        global w
        global keys
        self.experiments = None
        folders = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
        keys = ['']*len(folders)
        for i in range(len(folders)):
            try:
                with open(path+'/'+folders[i]+'/acqp','r') as RCSV:
                    trig=False
                    for line in RCSV.read().splitlines():
                        if trig:
                            opt=line
                            trig=False
                            break
                        if line.startswith('##$ACQ_scan_name='):
                            trig=True
                keys[i]=opt#,opt2)
            except:
                pass
        win_open = QtGui.QMainWindow()
        window = QtGui.QWidget()
        model = QtGui.QStandardItemModel(5,1)
        model.flags = lambda h: Qt.ItemIsEnabled | Qt.ItemIsSelectable
        model.setHorizontalHeaderLabels(['Experiments'])
        keys.sort()
        for row, text in enumerate(list(keys)):
            item = QtGui.QStandardItem(text)
            model.setItem(row, 0, item)
        layout = QtGui.QVBoxLayout(window)
        # filter proxy model
        filter_proxy_model = QtCore.QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(model)
        # table view
        table = QtGui.QTableView()
        table.setModel(filter_proxy_model)
        header = table.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(table)
        win_open.setCentralWidget(window)
        Open = QtGui.QAction("&Load",win_open)
        Open.setShortcut("Ctrl+L")
        Open.triggered.connect(lambda: self.load(table) )
        self.mainMenu = win_open.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&Data')
        self.fileMenu.addAction(Open)
        win_open.show()

class Main_Window:
    def __init__(self):
        global classes
        global palette
        self.classes = classes
        self.nsubimage=0
        self.nimage =0
        self.created_phantom=False
        self.app = QtGui.QApplication([])
        self.app.setStyle('Fusion')
        self.app.setPalette(palette)
        self.main_window = QtGui.QMainWindow()
        self.main_window.resize(800,800)
        self.add_menu()
        self.experiments = None
        self.start()
    def start(self):
        self.main_window.show()
        self.app.exec_()
    def add_menu(self):
        Open = QtGui.QAction("&Open",self.main_window)
        Open.setShortcut("Ctrl+O")
        Open.triggered.connect(self.open)
        Save = QtGui.QAction("&Save", self.main_window)
        Save.setShortcut("Ctrl+S")
        Save.triggered.connect(self.save_data)
        Saved = QtGui.QAction("&Saved data", self.main_window)
        Saved.triggered.connect(self.load_saved)
        Quit = QtGui.QAction("&Quit", self.main_window)
        Quit.triggered.connect(sys.exit)

        self.mainMenu = self.main_window.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(Open)
        self.fileMenu.addAction(Save)
        self.fileMenu.addAction(Saved)
        self.fileMenu.addAction(Quit)

    def open(self):
        global spam
        self.path = QtGui.QFileDialog.getExistingDirectory(self.main_window, 'Select path:','/home/pablojimenez' , QtGui.QFileDialog.ShowDirsOnly)
        if self.path=='':
            return
        Search_Window.__init__(self,self.path)
    def load(self,table):
        self.experiments = load(table)
        self.exp_names = load(table)
        folders_list = [s[s.find("(")+1:s.find(")")][1:] for s in self.experiments]
        self.experiments = folders_list
        win_open.close()
        self.load_tabs()
    def load_tabs(self):
        self.central_widget=MyTableWidget()
        self.main_window.setCentralWidget(self.central_widget)
        aux = self.path+'/'
        folders_list = self.experiments
        self.pb = ProgressBar() #1
        self.sequences = []
        for i in range(len(self.experiments)):
            self.sequences.append(PV_Parameters(aux+str(self.experiments[i])))
            if i!=0 and self.sequences[i].sequence_type!=self.sequences[i-1].sequence_type:
                QMessageBox.warning(self.main_window, 'Error', 'Sequences are different')
                self.pb.close()
            self.pb.setValue( int(7.*i/len(self.experiments))  )
            QApplication.processEvents()
        self.images = np.array([PV_Images(aux+str(f)).image_set for f in self.experiments ]   )
        if len(self.images.shape)==5:
            self.images=self.images[0]
        for i in range(1,len(self.images)):
            if self.images[i].shape!=self.images[i-1].shape:
                QMessageBox.warning(self.main_window, 'Error', 'Different number of sub-images')
                self.pb.close()
                return
        self.pb.setValue(7)
        QApplication.processEvents()
        Images_Widget.__init__(self)
        self.layout0 = QtGui.QGridLayout()
        self.central_widget.tab1.setLayout(self.layout0)
        Phantom_Widget.__init__(self,self.images[0][0])
        self.generate_graphix()
        self.d3_button = QPushButton()
        self.d3_button.setText('3D Image')
        self.d3_button.resize(100,32)     
        self.d3_button.clicked.connect(lambda: D3Image.__init__(self))
        self.layout1.addWidget(self.d3_button, 7, 3, 1, 1)
        self.anim_button = QPushButton()
        self.anim_button.setText('Animation')
        self.anim_button.resize(100,32)     
        self.anim_button.clicked.connect(lambda: Animation.__init__(self))
        self.layout1.addWidget(self.anim_button, 7, 4, 1, 1)
        self.pb.setValue(100)
        QApplication.processEvents()
        self.pb.close()
    def variable_list(self):
        window = QtGui.QWidget()
        model = QtGui.QStandardItemModel(5, 1)
        model.flags = lambda h: Qt.ItemIsEnabled | Qt.ItemIsSelectable
        model.setHorizontalHeaderLabels(['Variable'])
        for row, text in enumerate(list(self.sequences[0].obj.keys())):
            item = QtGui.QStandardItem(text)
            items=[]
            for i in range(len(self.sequences)):
                try:
                    items.append(self.sequences[i].obj[text])
                except:
                    print('Sequences have different parameters')
                    return
            #items = [f.obj[text] for f in self.sequences]
            if all(x == items[0] for x in items)==False:
                item.setForeground(QtGui.QColor('red'))
            model.setItem(row, 0, item)
        # filter proxy model
        filter_proxy_model = QtCore.QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(model)
        filter_proxy_model.setFilterKeyColumn(0) # first column
        
        # line edit for filtering
        layout = QtGui.QVBoxLayout(window)
        line_edit = QtGui.QLineEdit()
        line_edit.textChanged.connect(filter_proxy_model.setFilterRegExp)
        layout.addWidget(line_edit)
        # table view
        table = QtGui.QTableView()
        table.setModel(filter_proxy_model)
        header = table.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        table.doubleClicked.connect(lambda: self.selected_parameter(table) )
        table.resizeColumnToContents(True)
        layout.addWidget(table)

        self.layout1 = QtGui.QGridLayout()
        self.central_widget.tab2.setLayout(self.layout1)
        self.layout1.addWidget(window, 1, 1, 6, 2)   # button goes in upper-left
    def selected_parameter(self,table):
        index = table.selectedIndexes()[0]
        self.selected = table.model().data(index)
        self.show_parameters()
    def show_parameters(self):
        elements = [ (self.experiments[i],self.sequences[i].obj[self.selected]) for i in range(len(self.sequences))]
        elements =np.array(elements,dtype = [('Exp.',int),(self.selected, object)])
        self.w.setData(elements)
    def seq_parameters(self):
        self.w = pg.TableWidget()
        self.layout1.addWidget(self.w, 1, 3, 2, 2)
    def image_gallery(self):
        global imgss
        global itemss
        self.img_w = pg.ImageView()
        self.img_w.ui.roiBtn.hide()        
        plt.ioff()
        for i in range(self.images.shape[0]):
            for j in range(self.images.shape[1]):
                np.savetxt('Data/'+str(i)+'_'+str(j)+'.txt',self.images[i,j])
                fig = plt.figure()
                cur_axes = plt.gca()
                cur_axes.axes.get_xaxis().set_visible(False)
                cur_axes.axes.get_yaxis().set_visible(False)
                plt.imshow(self.images[i,j])
                plt.savefig('./Images/'+str(i)+'_'+str(j)+'.png')
                plt.close(fig)
                self.pb.setValue(7+int(45.*(i*self.images.shape[0]+j)/(self.images.shape[1]*self.images.shape[0])))
                QApplication.processEvents()

        number__ = self.images.shape[0]*self.images.shape[1]
        self.layout1.addWidget(self.img_w, 1, 5, 4, 5)
        imgss = np.array([['./Images/'+str(i)+'_'+str(j)+'.png' for i in range(self.images.shape[0])] for j in range(self.images.shape[1])]).T.flatten()
        itemss = ['i']*self.images.shape[0]*self.images.shape[1]
        for i in range(self.images.shape[0]):
            for j in range(self.images.shape[1]):
                itemss[i*self.images.shape[1]+j]='Img:'+str(i)+'. Sub:'+str(j)

        self.img_menu = ImageMenu()
        self.layout1.addWidget(self.img_menu, 3, 3, 4, 2)
        self.img_w.setImage(self.images[0,0])
        self.img_menu.tableviewA.doubleClicked.connect(self.selected_image)
    def selected_image(self):
        index=self.img_menu.tableviewA.selectedIndexes()[0]
        index=self.img_menu.tablemodel.items[index.row()]
        aux=index.split(':')
        self.nimage=int(aux[1][:-5])
        self.nsubimage=int(aux[2])
        self.img_w.setImage(self.images[self.nimage,self.nsubimage])
        for i in reversed(range(self.layout0.count())): 
            self.layout0.itemAt(i).widget().setParent(None)
        
        #self.phantom_image = self.images[self.nimage,self.nsubimage]
        #self.phantom_img1a.setImage(self.phantom_image)
        #self.phantom_aux.removeItem(self.phantom_img1a)
        #self.phantom_img1a = pg.ImageItem(self.phantom_image)
        #self.phantom_aux.addItem(self.phantom_img1a)
        Phantom_Widget.__init__(self,self.images[self.nimage,self.nsubimage])
    def generate_graphix(self):
        self.layout2 = QtGui.QGridLayout()
        self.central_widget.tab3.setLayout(self.layout2)
        self.pw_m = pg.GraphicsWindow()
        self.plot = self.pw_m.addPlot(title="Mean Signal in Experiments")
        self.plot_legend = self.plot.addLegend()
        self.pw_magn = [self.plot.plot(symbol='o',pen=self.colors[i],name='Region '+str(i+1)) for i in range(len(self.phantom_rois))]
        params = []
        self.function_dict = pd.DataFrame({}).to_dict(orient='list')
        for i in range(len(self.classes)):
            self.function_dict.update([( str(globals()[self.classes[i]]()) , self.classes[i] )])
            params.append({'name': str(globals()[self.classes[i]]()), 'type': 'action'})
        self.params = [{'name': 'Parameter', 'type': 'str', 'value': 'PVM_EchoTime'},{'name': 'Fit Curves', 'type': 'group','children':params}]
        self.p = Parameter.create(name='params', type='group', children=self.params)
        self.p.sigTreeStateChanged.connect(self.set_x_parameter)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.fit_parameters = ParameterTree()
        self.fited_parameters = pg.TableWidget()
        self.fitting_result = pg.GraphicsWindow()
        self.plot_fit = self.fitting_result.addPlot(title="Fitting results")
        self.plot_fit_legend = self.plot_fit.addLegend()
        self.layout2.addWidget(self.pw_m,0,0,3,1)
        self.layout2.addWidget(self.t,4,0,3,1)
        self.layout2.addWidget(self.fit_parameters,0,4,2,1)
        self.layout2.addWidget(self.fited_parameters,2, 4, 2, 1)
        self.layout2.addWidget(self.fitting_result,4, 4, 2, 1)
        self.x_data = [self.sequences[i].obj['PVM_EchoTime'] for i in range(len(self.sequences))]
        self.graphix()

    def graphix(self):
        self.magn=np.zeros((len(self.phantom_rois),self.images.shape[0]))
        for j in range(len(self.phantom_rois)):
            for i in range(self.images.shape[0]):
                auxiliar=np.array(self.phantom_rois[j].getArrayRegion(self.images[i,self.nsubimage], self.phantom_img1a, axes=(0, 1)))
                self.magn[j,i]=auxiliar[auxiliar!=0.].mean()
        for j in range(len(self.phantom_rois)):
            self.pw_magn[j].setData(x=self.x_data,y=self.magn[j,:])
    def set_x_parameter(self,param,changes):
        change_name = changes[0][0].name()
        if change_name=='Parameter':
            new = param.child('Parameter').value()
            self.x_data = [self.sequences[i].obj[new] for i in range(len(self.sequences))]
            self.graphix()
        else:
            self.curve_name = change_name
            self.fit_data(change_name)
    def phantom_change(self,param,changes):
        #param.child('Regions').value() = abs(param.child('Regions').value())
        new = param.child('Regions').value()
        if new ==len(self.phantom_rois):
            return
        #self.plot.scene().removeItem(self.plot_legend)
        self.plot_legend = self.plot.addLegend()
        if new<len(self.phantom_rois):
          for i in range(new,len(self.phantom_rois)):
            self.phantom_aux.removeItem(self.phantom_rois[i])
            self.pw_magn[i].clear()
            self.phantom_rois.pop(i)
            fff = [self.plot.removeItem(x) for x in self.pw_magn]
            self.phantom_p.child('Region '+str(i+1)).remove()

        if new>len(self.phantom_rois):
          for i in range(len(self.phantom_rois),new):
            fff = [self.plot.removeItem(x) for x in self.pw_magn]
            self.phantom_rois.append(pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True,pen=self.colors[(i)%len(self.colors)],removable=True))
            self.phantom_aux.addItem(self.phantom_rois[i])
            self.phantom_rois[i].sigRegionChanged.connect(self.phantom_update)
            self.phantom_rois[i].sigClicked.connect(lambda: self.region_histogram(self.phantom_rois[i]))
            self.phantom_rois[i].setAcceptedMouseButtons(QtCore.Qt.LeftButton)
            self.phantom_p.addChild({'name': 'Region '+str(i+1), 'type': 'color', 'value': self.colors[(i)%len(self.colors)],'readonly':True})

        self.pw_magn = [self.plot.plot(symbol='o',pen=self.colors[i],name='Region '+str(i+1)) for i in range(len(self.phantom_rois))]
        self.graphix()
    def phantom_update(self,roi):
        medio=np.zeros((len(self.phantom_rois)))
        desv=np.zeros((len(self.phantom_rois)))
        area=np.zeros((len(self.phantom_rois)))
        total=np.zeros((len(self.phantom_rois)))
        for i in range(len(self.phantom_rois)):
          auxiliar=np.array(self.phantom_rois[i].getArrayRegion(self.phantom_image, self.phantom_img1a, axes=(0, 1)))
          medio[i]=auxiliar[auxiliar!=0.].mean()
          desv[i]=auxiliar[auxiliar!=0.].std()
          area[i]=PolygonArea([[f[1].x(),f[1].y()] for f in self.phantom_rois[i].getSceneHandlePositions()])
          total[i]=auxiliar.sum()
          self.phantom_pw1.setData(x=list(range(len(medio))), y=medio)
          self.phantom_pw2.setData(x=list(range(len(area))), y=area)
          self.phantom_pw3.setData(x=list(range(len(desv))), y=desv,symbol='o')
          self.phantom_pw4.setData(x=list(range(len(total))), y=total,symbol='o')
        self.graphix()
    def console(self):
        namespace = {'pg': pg, 'np': np}
        text = "Python terminal (with numpy and pyqtgraph)"
        c = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
        self.layout1.addWidget(c, 5, 5, 2, 5)
    def fit_data(self,change_name):
        self.defined_f = globals()[self.function_dict[change_name]]()
        fit_params = []
        for i in range(len(self.defined_f.param_names)):
            fit_params.append({'name':self.defined_f.param_names[i],'type':self.defined_f.param_types[i],'suffix':' ['+self.defined_f.param_suffix[i]+']'})
        fit_vars = []
        for i in range(len(self.defined_f.var_names)):
            fit_vars.append({'name':self.defined_f.var_names[i],'type':'float','suffix':' ['+self.defined_f.var_suffix[i]+']'})
        pars = [{'name': 'Fit parameters','type':'group','children': fit_params},
        {'name': 'Seed values','type':'group','children': fit_vars},
        {'name':'Fit curve','type':'action'}]
        self.fit_p = Parameter.create(name='fit params', type='group', children=pars)
        try:
            self.fit_parameters.setParameters(self.fit_p, showTop=False)
        except:
            pass
        self.fit_p.sigTreeStateChanged.connect(self.fit_results)
    def fit_results(self,param,changes):
        change_name = changes[0][0].name()
        if 'Fit curve' in change_name:
            seed_pars=self.fit_p.getValues()['Fit parameters'][1]
            keys=list(seed_pars.keys())
            seed_pars = [seed_pars[f][0] for f in keys]
            result=[]
            self.defined_f.assign_params(seed_pars)
            seed_pars = self.fit_p.getValues()['Seed values'][1]
            keys=list(seed_pars.keys())
            seed_pars = [seed_pars[f][0] for f in keys]
            self.defined_f.assign_seeds(seed_pars)
            try:
                fff = [self.plot_fit.removeItem(x) for x in self.pw_magn_fit[0]]
                fff = [self.plot_fit.removeItem(x) for x in self.pw_magn_fit[1]]
            except:
                pass
            auxiliar = []
            for j in range(len(self.phantom_rois)):
                xdata = self.x_data
                ydata = self.magn[j,:]
                try:
                    popt, pcov = curve_fit(self.defined_f,xdata,ydata,p0=np.array(self.defined_f.seed),bounds=(0, np.inf))
                except:
                    popt=np.array(self.defined_f.seed)*0.
                result.append(tuple(popt))
                try:
                    self.plot_fit_legend = self.plot_fit.addLegend()
                except:
                    pass
                self.pw_magn_fit = [[self.plot_fit.plot(symbol='o',pen=self.colors[i],name='Region '+str(i+1))for i in range(len(self.phantom_rois))]]
                self.pw_magn_fit.append([self.plot_fit.plot(symbol='t1',pen=self.colors[i],name='Region '+str(i+1)+'. Fit')for i in range(len(self.phantom_rois))])
                x = np.linspace(min(self.x_data),max(self.x_data),70)
                y = self.defined_f(x,*popt)
                self.pw_magn_fit[0][j].setData(x=self.x_data,y=self.magn[j,:])
                self.pw_magn_fit[1][j].setData(x=x,y=y)
            results = np.array(result,dtype = [(g, float) for g in self.defined_f.var_names])
            self.fited_parameters.setData(results)
    def region_histogram(self,roi):
        auxiliar=np.array(roi.getArrayRegion(self.phantom_image, self.phantom_img1a, axes=(0, 1)))
        auxiliar=auxiliar.flatten()
        auxiliar=auxiliar[auxiliar!=0.]
        y,x = np.histogram(auxiliar, bins=np.linspace(auxiliar.min(), auxiliar.max(), 30))
        self.phantom_pw5.setData(x, y)
    def save_data(self):
        global wwin_open
        wwin_open = QtGui.QMainWindow()
        x= datetime.datetime.now()
        y = np.array([int(f) for f in self.experiments])
        params = [
            {'name': 'Experiment data', 'type': 'group', 'children': [
                {'name': 'Date', 'type': 'str', 'value': str(x.strftime("%c"))},
                {'name': 'Path', 'type': 'str', 'value': str(self.path)},
                {'name': 'Phantom', 'type': 'str', 'value': "-"},
                {'name': 'Experiments', 'type': 'group', 'children': [
                    {'name': 'From', 'type': 'int', 'value': y.min()},
                    {'name': 'To', 'type': 'int', 'value': y.max()},
                ]}]},
            ScalableGroup(name="Important parameters", children=[
                {'name': 'Param 1', 'type': 'str', 'value': "default param 1",'renamable': True ,'removable': True}
                ]),
            {'name': 'Extra info.', 'type': 'text', 'value': 'Some text...'},
            {'name': 'File name', 'type': 'str', 'value': ""},
            {'name': 'Save', 'type': 'action'}
            ]
        p = Parameter.create(name='params', type='group', children=params)
        p.children()[-1].sigTreeStateChanged.connect(lambda: self.save_file(p))
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        t.setWindowTitle('Data to save')
        wwin_open.setCentralWidget(t)
        wwin_open.show()
    def save_file(self,p):
        global elems
        elems = p.children()
        with open('./Saved/'+elems[-2].value()+'.txt','w') as f:
            for child in elems[0].children()[:-1]:
                f.write(child.name()+'\t'+child.type()+'\t'+str(child.value())+'\n')
            for child in elems[0].children()[-1].children():
                f.write(child.name()+'\t'+child.type()+'\t'+str(child.value())+'\n')
            for child in elems[1].children():
                f.write(child.name()+'\t'+child.type()+'\t'+str(child.value())+'\n')
            f.write(elems[2].name()+'\t'+elems[2].type()+'\t'+str(elems[2].value())+'\n')

    def load_saved(self):
        global wwin_open
        global data
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QtGui.QFileDialog.getOpenFileNames(self.main_window,'Select path:','./',"(*.txt)", options=options)
        data = [[g for g in line.split('\t')] for line in open(files[0],'r').read().split('\n')]
        wwin_open = QtGui.QMainWindow()
        childrens=[]
        for elems in data[5:-2]:
            childrens.append({'name':elems[0],'type':elems[1],'value':elems[2],'readonly': True})
        params = [
            {'name': 'Experiment data', 'type': 'group', 'children': [
                {'name': 'Date', 'type': 'str', 'value':  data[0][2],'readonly': True},
                {'name': 'Path', 'type': 'str', 'value':  data[1][2],'readonly': True},
                {'name': 'Phantom', 'type': 'str', 'value': data[2][2],'readonly': True},
                {'name': 'Experiments', 'type': 'group', 'children': [
                    {'name': 'From', 'type': 'int', 'value': int(data[3][2]),'readonly': True},
                    {'name': 'To', 'type': 'int', 'value': int(data[4][2]),'readonly': True},
                ]}]},
            ScalableGroup(name="Important parameters", children=childrens),
            {'name': 'Extra info.', 'type': 'text', 'value': data[-2][2],'readonly': True}
            ]
        p = Parameter.create(name='params', type='group', children=params)
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        t.setWindowTitle('Loaded data')
        wwin_open.setCentralWidget(t)
        wwin_open.show()
    def updateData(self):
        global anim_win
        if anim_win.open_win==False:
            return
        self.anim_img.setImage(self.images[self.i_update,self.nsubimage])
        self.i_update=(self.i_update+1)%len(self.images)
        QtCore.QTimer.singleShot(100, self.updateData)

x=Main_Window()