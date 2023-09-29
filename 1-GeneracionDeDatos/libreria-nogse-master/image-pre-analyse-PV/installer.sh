#!/bin/bash
export PATH="~/miniconda3/bin:$PATH"
echo y |conda  create -n Programita_NOGSE python=3.7.4
echo "environment has been created"
echo y |conda install -n Programita_NOGSE numpy
echo "numpy installed"
echo y |conda install -n Programita_NOGSE scipy
echo "scipypy installed"
echo y |conda install -n Programita_NOGSE -c dsdale24 pyqt5
echo "pyqt5 installed"
echo y | conda install -n Programita_NOGSE -c anaconda pandas
echo "pandas installed"
echo y | conda install -n Programita_NOGSE pyqtgraph
echo "pyqtgraph installed"
echo y | conda install -n Programita_NOGSE -c conda-forge pyperclip
echo "pyperclip installed"
echo y | conda install -n Programita_NOGSE -c conda-forge matplotlib
echo "matplotlib installed"
echo y | conda install -n Programita_NOGSE -c anaconda pyopengl
echo "opengl installed"
echo "all packages installed"
cd ~/Desktop/Librerias/
chmod u+x run.sh
exec sleep 1
chmod +x execution.desktop