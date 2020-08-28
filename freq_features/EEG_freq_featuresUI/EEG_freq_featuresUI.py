# EEG data analysis
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: july 2020
#
# Description: 
# This programm extract several features from preprocessed data tensor obtained via EEG_preprocessing 
# script. The resulting features are stored in another features tensor which is finally serialized 
# to .npy file.
#
# EntroPy was created and is maintained by Raphael Vallat.



import numpy as np
import scipy as sp
from scipy import signal

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def __init__(self, parent=None, **kwargs):
        self.path = False
        self.exctracted = False
        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(264, 320)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 211, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 40, 221, 31))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        self.buttonFile = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonFile.setObjectName("buttonFile")
        self.gridLayout.addWidget(self.buttonFile, 0, 1, 1, 1)
        self.comboFile = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboFile.setObjectName("comboFile")
        self.gridLayout.addWidget(self.comboFile, 0, 0, 1, 1)
        
        self.buttonExtract = QtWidgets.QPushButton(self.centralwidget)
        self.buttonExtract.setGeometry(QtCore.QRect(30, 150, 221, 51))
        self.buttonExtract.setObjectName("buttonExtract")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 120, 211, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.buttonSave = QtWidgets.QPushButton(self.centralwidget)
        self.buttonSave.setGeometry(QtCore.QRect(30, 210, 221, 51))
        self.buttonSave.setObjectName("buttonSave")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 264, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.showMessage('Ready')
        MainWindow.setStatusBar(self.statusbar)

        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")

        self.actionInformations = QtWidgets.QAction(MainWindow)
        self.actionInformations.setObjectName("actionInformations")

        self.menuMenu.addAction(self.actionQuit)
        self.menuMenu.addAction(self.actionInformations)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.actionQuit.triggered.connect(self.quit)

        self.actionInformations.triggered.connect(self.infos)

        self.buttonFile.clicked.connect(self.comboFile.clearEditText)
        self.buttonFile.clicked.connect(self.setPath)

        self.buttonExtract.clicked.connect(self.extract)
        self.buttonSave.clicked.connect(self.save)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EEG_freq_features"))
        self.label.setText(_translate("MainWindow", "Select the data tensor"))
        self.buttonFile.setText(_translate("MainWindow", "Browse .npy file"))
        self.buttonExtract.setText(_translate("MainWindow", "Extract spectrum and wavelets"))
        self.label_2.setText(_translate("MainWindow", "Frequency features extraction"))
        self.buttonSave.setText(_translate("MainWindow", "Save features tensors to .npy"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Esc"))
        self.actionInformations.setText(_translate("MainWindow", "Informations"))

    def quit(self):
        sys.exit()

    def infos(self):
        print("""   # EEG frequency features extractor
        #
        # Author: ROBALDO Axel for PI-Psy Institute
        # Date: july 2020
        #
        # Description: 
        # This programm extract the frequency sprectrum and Morlet continuous wavelets representation from 
        # preprocessed data tensor obtained via EEG_preprocessing script. The resulting features are stored 
        # in a 4D frequency features tensor and a 6D one which contain the wavelet representation of each
        # signal (2D pictures for each channel of each band on each epoch)
        # Tensors are finally serialized to .npy file.""")

    def setPath(self):
        self.pathNpy, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open Numpy Tensor File", "", "Npy Files (*.npy)")
        self.comboFile.addItem(self.pathNpy)
        self.path = True

    
    def extract(self):
        self.statusbar.showMessage('Features extraction, please wait...')
        
        if self.path == True:

            # useful constants----------------------------
            dt = 0.002
            fs = int(1/dt)
            epoch_duration = 0.5
            n_per_epochs = int(epoch_duration*fs)+1
            freq = np.linspace(1, fs/2, n_per_epochs)        

            w = 1.5                                             # wavelets omega parameter
            width = w*fs /(2*freq*np.pi)


            data_tensor = np.load(self.pathNpy)        # load input data tensor

            self.freq_feat_tensor = np.empty((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], n_per_epochs), dtype = np.float32) 
            self.wavelet_tensor = np.empty((data_tensor.shape[2], data_tensor.shape[1], data_tensor.shape[0], 1, n_per_epochs, n_per_epochs), dtype = np.float32)  #preallocate tensor of the good shape


            for epoch in np.ndindex(data_tensor.shape[0]):                  # for each epoch
                for wave in np.ndindex(data_tensor.shape[1]):               # for each wave
                    for channel in np.ndindex(data_tensor.shape[2]):        # for each channel
            
                        values = data_tensor[epoch[0], wave[0], channel[0], :]                      # take all the values (251) from the epoch
                        
                        # Fourier
                        self.freq_feat_tensor[epoch[0], wave[0], channel[0], :] = abs(sp.fft.fft(values))     # we process a 1D fft 

                        # Morlet wavelet            
                        self.wavelet_tensor[channel[0], wave[0], epoch[0], 0, :, :] = abs(sp.signal.cwt(values, sp.signal.morlet2, width, w = w))    # we get a (len(widths), len(data)) = (251,251) continuous wavelet tranform matrix

            print(self.freq_feat_tensor)
            print(self.wavelet_tensor)

            self.extracted = True
            self.statusbar.showMessage('Features extracted !')

        else:
            self.statusbar.showMessage('Select data tensor first')

    def save(self):
        if self.extracted == True:
            np.save("freq_features_tensor", self.freq_feat_tensor)             # serialize the tensors
            np.save("wavelets_tensor", self.wavelet_tensor)   
            self.statusbar.showMessage('Saved !')

        else:
            self.statusbar.showMessage('Extract data first')
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
